/*
 * ParaStation
 *
 * Copyright (C) 2025-2026 ParTec AG, Munich
 *
 * This file may be distributed under the terms of the Q Public License
 * as defined in the file LICENSE.QPL included in the packaging of this
 * file.
 */

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpitest.h"
#include "mpitestconf.h"

#define partitions 2
#define total_count 32
double buffer[total_count];
const double const_double_value = 3.14159265;

const char *test_function;

#define test_handle(function)                                           \
    test_function = #function ;                                         \
    do {                                                                \
        function;                                                       \
    } while (0);

#define test_assert(expression, string, ...)                            \
    if (!(expression)) {                                                \
        fprintf(stderr, "ERROR: %s " string " (%s:%d)\n",               \
                test_function,                                          \
                __VA_ARGS__, __FILE__, __LINE__);                       \
        errs++;                                                         \
    }

#define test_assert_key(key, flag)                                      \
    test_assert(flag, "did not find key \"%s\"", key);

/* This test program is to check the envisioned functionality of the proposed API extension
 * of the MPI interface by `MPIX_Register_compressor()`, working as a transparent plugin.
 *
 * The approach is to relocate both the registration function and the corresponding callback
 * functions to a loadable shared library, whose name is given as the first argv parameter.
 * By passing the info keys "compressor"/"compressor_plugin" the MPI library is requested to
 * search for a matching plugin that contains the "compressor_register" function, which is
 * supposed to register the corresponding callback.
 *
 * The test is passed successfully if the received double data matches the integer data that
 * was sent as compressed double values.
 */

int main(int argc, char *argv[])
{
    int errs = 0;
    MPI_Request request;
    MPI_Info info;
    int rank, size;
    MPI_Datatype dtype = MPI_DOUBLE;
    char value[MPI_MAX_INFO_VAL];
    int buflen = 0;
    char *key = NULL;
    int flag;

    MTest_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Info_create(&info);
    MPI_Info_set(info, "compressor", "pscompr");

    if (argc > 1) {
        if (!strcmp(argv[1], "double")) {
            dtype = MPI_DOUBLE;
        } else if (!strcmp(argv[1], "double_complex")) {
            dtype = MPI_DOUBLE_COMPLEX;
        }
    }

    if (argc > 2) {
        MPI_Info_set(info, "compressor", argv[2]);
    }

    if (argc > 3) {
        MPI_Info_set(info, "compressor_plugin", argv[3]);
    }

    if (argc > 4) {
        MPI_Info_set(info, "compressor_plugins", argv[4]);
    }

    if (rank == 0) {

        for (int j = 0; j < total_count; j++) {
            buffer[j] = const_double_value;
        }

        MPI_Psend_init(buffer, partitions, total_count / partitions, dtype, 1, 0,
                       MPI_COMM_WORLD, info, &request);

        MPI_Start(&request);

        for (int i = 0; i < partitions; i++) {
            MPI_Pready(i, request);
        }

        MPI_Wait(&request, MPI_STATUS_IGNORE);

    } else if (rank == 1) {

        MPI_Precv_init(buffer, partitions, total_count / partitions, dtype, 0, 0,
                       MPI_COMM_WORLD, info, &request);

        MPI_Start(&request);

        MPI_Wait(&request, MPI_STATUS_IGNORE);

        for (int j = 0; j < total_count; j++) {

            if ((int) buffer[j] != (int) const_double_value) {
                fprintf(stderr, "ERROR: Received %d (%f) at index %d but expected %d (%f)!\n",
                        (int) buffer[j], buffer[j], j, (int) const_double_value,
                        const_double_value);
                errs++;
            }
        }
    }

    MPI_Request_free(&request);

    /* Retrieve and store the compressor name for a final validity check. */
    key = "compressor";
    buflen = MPI_MAX_INFO_VAL;
    test_handle(MPI_Info_get_string(info, key, &buflen, value, &flag));
    test_assert_key(key, flag);
    char *compr_name = flag ? strdup(value) : NULL;

    MPI_Info_free(&info);

    /* Check if a matching compressor plugin has actually been registered (implicitly) up to now. */
    test_handle(MPI_Info_create_env(0, NULL, &info));
    test_assert(info != MPI_INFO_NULL, "returned MPI_INFO_NULL (%p).", &info);
    key = "compressor";
    buflen = MPI_MAX_INFO_VAL;
    test_handle(MPI_Info_get_string(info, key, &buflen, value, &flag));
    test_assert_key(key, flag);
    if (flag && compr_name) {
        test_assert(!strcmp(value, compr_name),
                    "returned value \"%s\" that is not the right compressor", value);
    }

    MTest_Finalize(errs);
    return MTestReturnValue(errs);
}
