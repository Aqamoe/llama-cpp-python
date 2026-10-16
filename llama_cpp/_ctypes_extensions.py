from __future__ import annotations

import sys
import os
import ctypes
import functools
import pathlib
import importlib.metadata
from ctypes.util import find_library
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Union,
    Optional,
    TYPE_CHECKING,
    TypeVar,
    Generic,
)
from typing_extensions import TypeAlias

def _version_at_least(version: str) -> bool:
    """Check whether installed llama-cpp-python version meets requirement."""
    try:
        current = importlib.metadata.version("llama-cpp-python")
        from packaging.version import Version
        return Version(current) >= Version(version)
    except Exception:
        return False

def _format_library_dir_contents(base_paths: list[pathlib.Path]) -> str:
    """Format directory contents for diagnostics after library loading fails."""
    sections = []

    for base_path in base_paths:
        p = pathlib.Path(base_path)

        if not p.exists():
            sections.append(f"{p}: <not found>")
            continue

        if not p.is_dir():
            sections.append(f"{p}: <not a directory>")
            continue

        try:
            # Only list files when reporting a final loading failure.
            files = sorted(x.name for x in p.iterdir())
        except Exception as e:
            sections.append(f"{p}: <failed to list: {e}>")
            continue

        if files:
            sections.append(
                f"{p}:\n"
                + "\n".join(f"  - {name}" for name in files)
            )
        else:
            sections.append(f"{p}: <empty>")

    return "\n".join(sections)

def _win32_error_text(code: int) -> str:
    """把 Win32 错误码转成系统文案（中文系统给中文）。"""
    try:
        buf = ctypes.create_unicode_buffer(512)
        got = ctypes.windll.kernel32.FormatMessageW(
            0x00001000,  # FORMAT_MESSAGE_FROM_SYSTEM
            None, code, 0, buf, 512, None)
        return buf.value.strip() if got else ""
    except Exception:
        return ""


# 已知的 Win32 加载失败语义 —— 把 "Failed to load dynlib/dll" 变成可定位的原因。
# 注：126/127 是 Windows 上最常见的"间接依赖"失败：文件本身存在，但它的某个
# 依赖不可用（缺文件 → 126；缺导出符号/版本过旧 → 127），报错文案里那个
# "was not found" 指的是文件的依赖，不是文件自己。
_WIN32_LOAD_HINTS = {
    126: "依赖的 DLL 缺失（模块找不到）",
    127: "文件找到了但缺少所需导出函数（版本过旧或被替换）",
    193: "架构不匹配（例如 64 位进程里加载了 32 位 DLL）",
    5: "被安全软件/策略拦截（拒绝访问）",
    225: "被安全软件拦截（文件被标记为病毒）",
    1114: "DLL 已找到但初始化失败（依赖组件损坏）",
}

# 需要额外探测的系统依赖：llama.cpp 的 Vulkan 后端的导入链是
# ggml.dll → ggml-vulkan.dll → 系统的 vulkan-1.dll，只要系统里那份
# vulkan-1.dll 缺失/过旧/被拦，ggml.dll 就会整体加载失败（与是否走 GPU 无关）。
_WIN32_SYSTEM_DEPS = ("vulkan-1.dll",)


def _win32_load_error_code(path: str) -> int:
    """用 LoadLibraryExW 复现一次失败，取回真实 Win32 错误码。

    必要性：PyInstaller 冻结包里 ctypes 抛出的 OSError 往往不带 winerror
    （只剩 "Failed to load dynlib/dll ... frozen" 这条通用文案），而普通
    Python 下是带的。自己调一次 LoadLibraryExW 才能拿到 126/127/193/1114
    这类能定位问题的错误码。
    """
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.LoadLibraryExW.restype = ctypes.c_void_p
        kernel32.LoadLibraryExW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p,
                                            ctypes.c_uint32]
        if kernel32.LoadLibraryExW(path, None, 0):
            return 0  # 复现不出来（上一步失败是瞬时的），不猜测
        return ctypes.get_last_error() or 0
    except Exception:  # noqa: BLE001 - 诊断路径
        return 0


def _win32_format_error(exc: BaseException, lib_path=None) -> str:
    """把 CDLL 抛出的 OSError 变成带 Win32 错误码/文案/解读的一行。"""
    if sys.platform != "win32":
        return str(exc)
    code = getattr(exc, "winerror", None) or 0
    if not code and lib_path:
        code = _win32_load_error_code(str(lib_path))
    if not code:
        return str(exc)
    parts = [f"winerror={code}"]
    text = _win32_error_text(code)
    if text:
        parts.append(text)
    hint = _WIN32_LOAD_HINTS.get(code, "")
    if hint:
        parts.append(hint)
    return f"{exc} [{'；'.join(parts)}]"


def _win32_probe_dll(name: str) -> str:
    """探测某个 DLL 在当前进程里会被解析到哪个文件（解析不到就报错误码）。

    仅在加载失败的诊断路径上调用：用来回答"到底缺了什么"。探测会真实加载
    该 DLL，但此时 import 已经注定失败，没有副作用需要考虑。
    """
    if sys.platform != "win32":
        return ""
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.LoadLibraryExW.restype = ctypes.c_void_p
        kernel32.LoadLibraryExW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p,
                                            ctypes.c_uint32]
        handle = kernel32.LoadLibraryExW(name, None, 0)
        if not handle:
            code = ctypes.get_last_error()
            text = _win32_error_text(code)
            return (f"{name}: 无法加载 (winerror={code}"
                    + (f" {text}" if text else "") + ")")
        buf = ctypes.create_unicode_buffer(1024)
        kernel32.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p,
                                                ctypes.c_uint32]
        kernel32.GetModuleFileNameW(handle, buf, 1024)
        path = buf.value
        size = ""
        try:
            size = f", {os.path.getsize(path)} 字节"
        except OSError:
            pass
        return f"{name}: 已解析为 {path}{size}"
    except Exception as e:  # noqa: BLE001 - 诊断路径，任何异常都不该再抛
        return f"{name}: 探测失败 ({e})"


def _win32_system_dep_probe() -> str:
    """把 llama.cpp 依赖的系统 DLL 解析情况拼成一段可读文本。"""
    if sys.platform != "win32":
        return ""
    lines = [f"  - {_win32_probe_dll(name)}" for name in _WIN32_SYSTEM_DEPS]
    return "\n".join(lines)


# Load the library
def load_shared_library(lib_base_name: str, base_paths: Union[pathlib.Path, list[pathlib.Path]]):
    if isinstance(base_paths, pathlib.Path):
        base_paths = [base_paths]

    lib_names = []

    if sys.platform.startswith("linux") or sys.platform.startswith("freebsd"):
        lib_names = [f"lib{lib_base_name}.so"]

        base_paths.extend([
            "/usr/local/lib",
            "/usr/lib",
            "/usr/lib64",
        ])

    elif sys.platform == "darwin":
        lib_names = [
            f"lib{lib_base_name}.dylib",
            f"lib{lib_base_name}.so",
        ]

        base_paths.extend([
            "/usr/local/lib",
            "/opt/homebrew/lib",
            "/usr/lib",
        ])

    elif sys.platform == "win32":
        lib_names = [
            f"{lib_base_name}.dll",
            f"lib{lib_base_name}.dll",
        ]
    else:
        raise RuntimeError("Unsupported platform")

    cdll_args = dict()  # type: ignore

    # Add the library directory to the DLL search path on Windows (if needed)
    if sys.platform == "win32":

        # Add CUDA runtime DLL directories if CUDA is available.
        if "CUDA_PATH" in os.environ:
            cuda_path = os.environ["CUDA_PATH"]
            sub_dirs_to_add = [
                "bin",
                os.path.join("bin", "x64"),  # CUDA 13.0+
                "lib",
                os.path.join("lib", "x64")
            ]
            for sub_dir in sub_dirs_to_add:
                full_path = os.path.join(cuda_path, sub_dir)
                if os.path.exists(full_path):
                    os.add_dll_directory(full_path)

        # Add HIP runtime DLL directories when HIP backend is available.
        if "HIP_PATH" in os.environ:
            hip_path = os.environ["HIP_PATH"]
            for sub_dir in ["bin", "lib"]:
                full_path = os.path.join(hip_path, sub_dir)
                if os.path.exists(full_path):
                    os.add_dll_directory(full_path)

        # Add Vulkan SDK DLL directories when Vulkan backend is enabled.
        if "VULKAN_SDK" in os.environ:
            vulkan_sdk = os.environ["VULKAN_SDK"]
            for sub_dir in ["Bin", "Lib"]:
                full_path = os.path.join(vulkan_sdk, sub_dir)
                if os.path.exists(full_path):
                    os.add_dll_directory(full_path)

        # Add package-provided library directories.
        #
        # The paths are added in reverse order intentionally.
        # This ensures that the first entry in base_paths gets prepended
        # to PATH last, making it the highest priority search location.
        #
        # Example:
        #   base_paths = [
        #       package/lib,
        #       package/bin,
        #   ]
        #
        # After reversed iteration:
        #   PATH = package/lib;package/bin;...
        for base_path in reversed(base_paths):
            p = pathlib.Path(base_path)
            if p.exists() and p.is_dir():
                os.add_dll_directory(str(p))
                os.environ["PATH"] = str(p) + os.pathsep + os.environ["PATH"]

        cdll_args["winmode"] = ctypes.RTLD_GLOBAL

    errors = []

    # First, try to find an available library through the system
    lib_path = find_library(lib_base_name)
    if lib_path:
        try:
            lib = ctypes.CDLL(lib_path, **cdll_args)
            print(f"[llama-cpp-python].find_library: loaded library from {lib_path}")
            return lib
        except Exception as e:
            errors.append(f"{lib_path}: {_win32_format_error(e, lib_path)}")

    # Then fallback to manually checking the list of paths.
    for base_path in base_paths:
        for lib_name in lib_names:
            lib_path = pathlib.Path(base_path) / lib_name

            if lib_path.exists():
                try:
                    lib = ctypes.CDLL(str(lib_path), **cdll_args)
                    print(f"[llama-cpp-python].provided_path: loaded library from {lib_path}")
                    return lib
                except Exception as e:
                    errors.append(f"{lib_path}: {_win32_format_error(e, lib_path)}")

    # Include directory contents only in the failure path to avoid extra work during successful imports.
    message = (
        f"Failed to load '{lib_base_name}' from {base_paths}\n"
        + "\n".join(errors)
        + "\nLibrary search path contents:\n"
        + _format_library_dir_contents(base_paths)
    )
    # Windows 上再多加一段"系统依赖解析结果"：绝大多数加载失败其实是某个
    # 间接依赖（如 vulkan-1.dll）不可用，这段能把真凶直接写进日志。
    if sys.platform == "win32":
        probe = _win32_system_dep_probe()
        if probe:
            message += "\nSystem dependency probe:\n" + probe
    raise RuntimeError(message)


# ctypes sane type hint helpers
#
# - Generic Pointer and Array types
# - PointerOrRef type with a type hinted byref function
#
# NOTE: Only use these for static type checking not for runtime checks
# no good will come of that

if TYPE_CHECKING:
    CtypesCData = TypeVar("CtypesCData", bound=ctypes._CData)  # type: ignore

    CtypesArray: TypeAlias = ctypes.Array[CtypesCData]  # type: ignore

    CtypesPointer: TypeAlias = ctypes._Pointer[CtypesCData]  # type: ignore

    CtypesVoidPointer: TypeAlias = ctypes.c_void_p

    class CtypesRef(Generic[CtypesCData]):
        pass

    CtypesPointerOrRef: TypeAlias = Union[
        CtypesPointer[CtypesCData], CtypesRef[CtypesCData]
    ]

    CtypesFuncPointer: TypeAlias = ctypes._FuncPointer  # type: ignore

F = TypeVar("F", bound=Callable[..., Any])


def ctypes_function_for_shared_library(lib: ctypes.CDLL):
    """Create a decorator used to bind typed Python declarations to C symbols.

    The returned decorator accepts either a single exported symbol name or an
    iterable of ABI-compatible aliases. When aliases are provided, they are
    checked in order and the first available symbol is selected.
    """

    def ctypes_function(
        name: Union[str, Iterable[str]],
        argtypes: List[Any],
        restype: Any,
        enabled: bool = True,
        required: bool = True,
    ):
        """Bind a Python declaration to one of the requested C symbols.

        Args:
            name: A symbol name or an ordered iterable of compatible aliases.
            argtypes: The ctypes argument types assigned to the C function.
            restype: The ctypes return type assigned to the C function.
            enabled: Return the original Python declaration when disabled.
            required: Raise if symbol is missing. If False, create a runtime unavailable stub.

        Raises:
            ValueError: If no symbol names are provided.
            AttributeError: If none of the requested symbols exist in the
                shared library.
        """
        symbol_names = (name,) if isinstance(name, str) else tuple(name)

        if not symbol_names:
            raise ValueError("At least one shared library symbol name is required")

        def decorator(f: F) -> F:
            if not enabled:
                return f

            for symbol_name in symbol_names:
                try:
                    func = getattr(lib, symbol_name)
                except AttributeError:
                    continue
                # Validate ctypes argument declarations before assigning them.
                # ctypes requires every argtype to provide from_param().
                for index, argtype in enumerate(argtypes):
                    if not hasattr(argtype, "from_param"):
                        raise TypeError(
                            "Invalid ctypes argument type:\n"
                            f"  function: {f.__name__}\n"
                            f"  symbol: {symbol_name}\n"
                            f"  arg index: {index}\n"
                            f"  arg type: {argtype!r}\n"
                            f"  expected: a ctypes type with from_param()"
                        )

                func.argtypes = argtypes
                func.restype = restype
                functools.update_wrapper(func, f)

                # Preserve the actual exported symbol selected at runtime for
                # diagnostics, especially when ABI aliases are being used.
                func.__ctypes_symbol_name__ = symbol_name
                return func

            message = (
                "None of the shared library symbols were found: "
                + ", ".join(symbol_names)
            )

            if required:
                raise AttributeError(message)

            # Optional extension API.
            # Keep import working when the symbol is unavailable.
            print(
                "[llama-cpp-python].ctypes_function: WARNING! optional API unavailable\n"
                f"  symbols: {', '.join(symbol_names)}\n"
                f"  library: {getattr(lib, '_name', '<unknown>')}"
            )

            def unavailable(*args, **kwargs):
                raise RuntimeError(
                    "This llama.cpp extension API is unavailable.\n"
                    f"Required symbol(s): {', '.join(symbol_names)}\n"
                    f"Library: {getattr(lib, '_name', '<unknown>')}"
                )

            functools.update_wrapper(unavailable, f)

            # Mark unavailable extension API.
            unavailable.__ctypes_symbol_name__ = None
            unavailable.__ctypes_optional__ = True

            return unavailable

        return decorator

    return ctypes_function


def _byref(obj: CtypesCData, offset: Optional[int] = None) -> CtypesRef[CtypesCData]:
    """Type-annotated version of ctypes.byref"""
    ...


byref = _byref if TYPE_CHECKING else ctypes.byref
