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
    explicit asThreadTestCuda();

    ~asThreadTestCuda() override = default;

    ExitCode Entry() override;

    void OnExit() override;

  protected:

  private:

};


asThreadTestCuda::asThreadTestCuda()
        : asThread()
{
}

wxThread::ExitCode asThreadTestCuda::Entry()
{
    CudaProcessSum();

    return 0;
}

TEST(Cuda, UseInSingleThread)
{
    wxCriticalSection threadCS;

    auto* thread = new asThreadTestCuda();

    while (true) {
        {
            wxCriticalSectionLocker enter(threadCS);
            if (!thread)
                break;
        }
        wxMilliSleep(10);
    }
}

TEST(Cuda, UseInTwoThreads)
{
    wxCriticalSection threadCS;

    auto *thread1 = new asThreadTestCuda();
    auto *thread2 = new asThreadTestCuda();

    while (true) {
        {
            wxCriticalSectionLocker enter(threadCS);
            if (!thread1 && !thread2)
                break;
        }
        wxMilliSleep(10);
    }
}

#endif