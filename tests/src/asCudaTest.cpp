/*
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS HEADER.
 *
 * The contents of this file are subject to the terms of the
 * Common Development and Distribution License (the "License").
 * You may not use this file except in compliance with the License.
 *
 * You can read the License at http://opensource.org/licenses/CDDL-1.0
 * See the License for the specific language governing permissions
 * and limitations under the License.
 *
 * When distributing Covered Code, include this CDDL Header Notice in
 * each file and include the License file (licence.txt). If applicable,
 * add the following below this CDDL Header, with the fields enclosed
 * by brackets [] replaced by your own identifying information:
 * "Portions Copyright [year] [name of copyright owner]"
 *
 * The Original Software is AtmoSwing.
 * The Original Software was developed at the University of Lausanne.
 * All Rights Reserved.
 *
 */

/*
 * Portions Copyright 2019 Pascal Horton, University of Bern.
 */

#ifdef USE_CUDA

#include "asThread.h"
#include "asCuda.cuh"
#include "gtest/gtest.h"

class asThreadTestCuda
    : public asThread
{
  public:
    explicit asThreadTestCuda(const wxString &test);

    ~asThreadTestCuda() override = default;

    ExitCode Entry() override;

  protected:

  private:
    wxString m_test;

};


asThreadTestCuda::asThreadTestCuda(const wxString &test)
        : asThread(),
          m_test(test) {
}

wxThread::ExitCode asThreadTestCuda::Entry()
{
    if (m_test.IsSameAs("simple")) {
        CudaProcessSum();
    } else if (m_test.IsSameAs("streams")) {
        CudaProcessSumWithStreams();
    } else {
        wxLogError(_("CUDA test name not correctly defined."));
    }

    return 0;
}

TEST(Cuda, UseInSingleThread)
{
    wxCriticalSection threadCS;

    auto* thread = new asThreadTestCuda("simple");

    ThreadsManager().AddThread(thread);
    ThreadsManager().Wait(asThread::Undefined);
}

TEST(Cuda, UseInTwoThreads)
{
    auto *thread1 = new asThreadTestCuda("simple");
    auto *thread2 = new asThreadTestCuda("simple");

    ThreadsManager().AddThread(thread1);
    ThreadsManager().AddThread(thread2);
    ThreadsManager().Wait(asThread::Undefined);
}

TEST(Cuda, UseInManyThreads)
{
    for (int i = 0; i < 100; ++i) {
        auto thread = new asThreadTestCuda("simple");
        ThreadsManager().AddThread(thread);
    }

    ThreadsManager().Wait(asThread::Undefined);
}

TEST(Cuda, UseInManyThreadsWithStreams)
{
    for (int i = 0; i < 100; ++i) {
        auto thread = new asThreadTestCuda("streams");
        ThreadsManager().AddThread(thread);
    }

    ThreadsManager().Wait(asThread::Undefined);
}

#endif