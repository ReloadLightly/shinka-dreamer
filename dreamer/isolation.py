"""Small Linux execution boundary: Landlock filesystem + seccomp syscalls.

Applied before exec of candidate source; fail closed if unavailable. No namespaces
or global settings needed. Not protection against kernel vulnerabilities or covert
timing channels. See docs/protocol.md for the supported computation environment.
"""
import ctypes
import errno
import os
import resource


def restrict():
    libc = ctypes.CDLL(None, use_errno=True)
    sec = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0):  # no_new_privs
        raise OSError(ctypes.get_errno(), "no_new_privs")
    abi = libc.syscall(444, 0, 0, 1)
    if abi < 1:
        raise RuntimeError("Landlock unavailable; refusing candidate execution")

    class Ruleset(ctypes.Structure):
        _fields_ = [("handled_access_fs", ctypes.c_uint64)]

    class PathRule(ctypes.Structure):
        _pack_ = 1
        _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]

    mask = (1 << (15 if abi >= 3 else 14 if abi >= 2 else 13)) - 1
    ruleset = Ruleset(mask)
    fd = libc.syscall(444, ctypes.byref(ruleset), ctypes.sizeof(ruleset), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset")
    # Only system Python's standard library is importable. No home/site packages,
    # evaluator, repository, /proc, credentials or assessment artifacts.
    for path in ("/usr/lib/python3.10",):
        pfd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        rule = PathRule((1 << 2) | (1 << 3), pfd)  # READ_FILE, READ_DIR
        if libc.syscall(445, fd, 1, ctypes.byref(rule), 0):
            raise OSError(ctypes.get_errno(), "landlock_add_rule")
        os.close(pfd)
    if libc.syscall(446, fd, 0):
        raise OSError(ctypes.get_errno(), "landlock_restrict_self")
    os.close(fd)
    sec.seccomp_init.argtypes = [ctypes.c_uint32]
    sec.seccomp_init.restype = ctypes.c_void_p
    sec.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    sec.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    sec.seccomp_load.argtypes = [ctypes.c_void_p]
    sec.seccomp_release.argtypes = [ctypes.c_void_p]
    ctx = sec.seccomp_init(0x00050000 | errno.EPERM)
    allowed = "read write close fstat newfstatat stat lstat lseek openat getdents64 readlink readlinkat access faccessat mmap mprotect munmap mremap madvise brk rt_sigaction rt_sigprocmask rt_sigreturn sigaltstack futex clock_gettime clock_nanosleep nanosleep gettimeofday time getrandom getpid gettid getuid geteuid getgid getegid uname sched_getaffinity fcntl ioctl dup dup2 dup3 exit exit_group restart_syscall"
    for name in allowed.split():
        nr = sec.seccomp_syscall_resolve_name(name.encode())
        if nr >= 0 and sec.seccomp_rule_add(ctx, 0x7FFF0000, nr, 0):
            raise RuntimeError("seccomp rule failed")
    resource.setrlimit(resource.RLIMIT_AS, (192 * 1024**2,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if sec.seccomp_load(ctx):
        raise RuntimeError("seccomp load failed")
    sec.seccomp_release(ctx)
