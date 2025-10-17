#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <sys/wait.h>
#include <errno.h>
#include <poll.h>
#include <fcntl.h>

typedef int (*execve_t)(const char *, char *const[], char *const[]);

static void safe_write_all(int fd, const void *buf, size_t len) {
    const unsigned char *p = (const unsigned char *)buf;
    while (len > 0) {
        ssize_t w = write(fd, p, len);
        if (w < 0) {
            if (errno == EINTR) continue;
            break;
        }
        p += (size_t)w;
        len -= (size_t)w;
    }
}

int execve(const char *filename, char *const argv[], char *const envp[]) {
    static execve_t real_execve = NULL;
    if (!real_execve)
        real_execve = (execve_t)dlsym(RTLD_NEXT, "execve");

    const char *logfile = getenv("LOGGER_FILE");
    if (!logfile) logfile = "/agent_output/tmp.log";

    // Prepare pipes to tee stdout and stderr without altering external flow
    int out_pipe[2];
    int err_pipe[2];
    if (pipe(out_pipe) == -1 || pipe(err_pipe) == -1) {
        // Fallback: if we cannot set up, just exec without logging to avoid breaking behavior
        return real_execve(filename, argv, envp);
    }

    // Duplicate current stdout/stderr so tee process can forward to originals
    int stdout_copy = dup(STDOUT_FILENO);
    int stderr_copy = dup(STDERR_FILENO);
    if (stdout_copy == -1 || stderr_copy == -1) {
        // Cleanup and fallback
        close(out_pipe[0]); close(out_pipe[1]);
        close(err_pipe[0]); close(err_pipe[1]);
        if (stdout_copy != -1) close(stdout_copy);
        if (stderr_copy != -1) close(stderr_copy);
        return real_execve(filename, argv, envp);
    }

    pid_t tee_pid = fork();
    if (tee_pid < 0) {
        // Fork failed; cleanup and fallback
        close(out_pipe[0]); close(out_pipe[1]);
        close(err_pipe[0]); close(err_pipe[1]);
        close(stdout_copy); close(stderr_copy);
        return real_execve(filename, argv, envp);
    }

    if (tee_pid == 0) {
        // Tee/logging helper process
        // Close write ends; we'll read from read ends
        close(out_pipe[1]);
        close(err_pipe[1]);

        FILE *f = fopen(logfile, "a");
        time_t t = time(NULL);
        char ts[64];
        struct tm *lt = localtime(&t);
        if (lt) strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", lt);
        else strncpy(ts, "unknown-time", sizeof(ts));

        if (f) {
            fprintf(f, "\n[%s] PID %d ran: %s", ts, getppid(), filename);
            for (int i = 1; argv && argv[i]; i++) {
                fprintf(f, " %s", argv[i]);
            }
            fprintf(f, "\n--- Output (tee) ---\n");
            fflush(f);
        }

        // Deduplication buffers for each stream to avoid logging identical consecutive lines
        char last_stdout[4096] = {0};
        char last_stderr[4096] = {0};
        size_t last_stdout_len = 0;
        size_t last_stderr_len = 0;
        int current_pid = getppid();

        struct pollfd pfds[2];
        pfds[0].fd = out_pipe[0];
        pfds[0].events = POLLIN;
        pfds[1].fd = err_pipe[0];
        pfds[1].events = POLLIN;

        int open_count = 2;
        char buf[4096];
        while (open_count > 0) {
            int pr = poll(pfds, 2, -1);
            if (pr < 0) {
                if (errno == EINTR) continue;
                break;
            }

            for (int i = 0; i < 2; i++) {
                if (pfds[i].fd == -1) continue;
                if (pfds[i].revents & (POLLIN)) {
                    ssize_t n = read(pfds[i].fd, buf, sizeof(buf));
                    if (n > 0) {
                        // First: forward to original destination (preserve external flow)
                        int target_fd = (i == 0) ? stdout_copy : stderr_copy;
                        safe_write_all(target_fd, buf, (size_t)n);

                        // Second: check for deduplication and write to logfile with PID tag
                        if (f) {
                            char *last_buf = (i == 0) ? last_stdout : last_stderr;
                            size_t *last_len = (i == 0) ? &last_stdout_len : &last_stderr_len;
                            
                            // Check if this is identical to the last output from this stream
                            if (n == *last_len && memcmp(buf, last_buf, n) == 0) {
                                // Skip logging identical consecutive output
                                continue;
                            }
                            
                            // Update last buffer
                            if (n <= sizeof(last_stdout)) {
                                memcpy(last_buf, buf, n);
                                *last_len = n;
                            }
                            
                            // Write to log with PID tag
                            if (i == 0) {
                                fprintf(f, "[{%d} STDOUT] ", current_pid);
                            } else {
                                fprintf(f, "[{%d} STDERR] ", current_pid);
                            }
                            fwrite(buf, 1, (size_t)n, f);
                            fflush(f);
                        }
                    } else if (n == 0) {
                        // EOF on this stream
                        close(pfds[i].fd);
                        pfds[i].fd = -1;
                        open_count--;
                    }
                }
                if (pfds[i].revents & (POLLERR | POLLHUP | POLLNVAL)) {
                    if (pfds[i].fd != -1) close(pfds[i].fd);
                    pfds[i].fd = -1;
                    open_count--;
                }
            }
        }

        if (f) {
            fprintf(f, "\n--- End ---\n");
            fclose(f);
        }

        // Close duplicates
        close(stdout_copy);
        close(stderr_copy);
        _exit(0);
    }

    // Parent (will become the target process). Redirect stdout/stderr to pipes' write ends
    close(out_pipe[0]);
    close(err_pipe[0]);
    dup2(out_pipe[1], STDOUT_FILENO);
    dup2(err_pipe[1], STDERR_FILENO);
    close(out_pipe[1]);
    close(err_pipe[1]);

    // The tee child holds stdout_copy/stderr_copy and will forward. We must close our copies here.
    close(stdout_copy);
    close(stderr_copy);

    // Finally, exec the real program in this process to preserve execve semantics
    return real_execve(filename, argv, envp);
}