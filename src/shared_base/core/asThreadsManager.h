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
 * Portions Copyright 2008-2013 Pascal Horton, University of Lausanne.
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#ifndef AS_THREADS_MANAGER_H
#define AS_THREADS_MANAGER_H

#include "asIncludes.h"

class asThread;

class asThreadsManager : public wxObject {
   public:
    asThreadsManager();

    ~asThreadsManager() override;

    void Init();

    void OnClose(wxCloseEvent &);

    bool AddThread(asThread *thread);

    void Wait(int type);

    bool HasFreeThread(int type);

    void WaitForFreeThread(int type);

    void SetNull(wxThreadIdType id);

    bool CleanArray();

    void PauseAll();

    void ResumeAll();

    int GetTotalThreadsNb();

    int GetRunningThreadsNb(int type = -1);

    int GetFreeDevice(int devicesNb);

    int GetAvailableThreadsNb();

    bool Cancelled() {
        wxCriticalSectionLocker lock(m_critSectionManager);
        return m_cancelled;
    }

    void Cancel() {
        wxCriticalSectionLocker lock(m_critSectionManager);
        m_cancelled = true;
    }

    wxCriticalSection &CritSectionNetCDF() {
        return m_critSectionNetCDF;
    }

    wxCriticalSection &CritSectionGrib() {
        return m_critSectionGrib;
    }

    wxCriticalSection &CritSectionConfig() {
        return m_critSectionConfig;
    }

    wxCriticalSection &CritSectionPreloadedData() {
        return m_critSectionPreloadedData;
    }

    wxSemaphore &SemAllDone() {
        return m_semAllDone;
    }

    bool GetWaitingUntilAllDone() {
        return m_waitingUntilAllDone;
    }

    void SetWaitingUntilAllDone(bool val) {
        m_waitingUntilAllDone = val;
    }

   protected:
   private:
    int m_idCounter;
    std::vector<asThread *> m_threads;
    wxCriticalSection m_critSectionManager;
    wxCriticalSection m_critSectionPreloadedData;
    wxCriticalSection m_critSectionNetCDF;
    wxCriticalSection m_critSectionGrib;
    wxCriticalSection m_critSectionConfig;
    wxSemaphore m_semAllDone;
    bool m_waitingUntilAllDone;
    bool m_cancelled;
    int m_maxThreadsNb;
    int m_priority;
};

#endif
