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

#include "asThreadsManager.h"

#include "asThread.h"

// Safe: Critical section defined within
asThreadsManager::asThreadsManager()
    : _idCounter(-1),
      _waitingUntilAllDone(true),
      _cancelled(false),
      _maxThreadsNb(-1),
      _priority(-1) {}

asThreadsManager::~asThreadsManager() {
    if (GetRunningThreadsNb() > 0) {
        CleanArray();
    }
}

void asThreadsManager::Init() {
    _critSectionManager.Enter();

    wxConfigBase* pConfig = wxFileConfig::Get();
    _maxThreadsNb = wxThread::GetCPUCount();
    if (_maxThreadsNb == -1) _maxThreadsNb = 2;
    pConfig->Read("/Processing/ThreadsNb", &_maxThreadsNb, _maxThreadsNb);
    _priority = pConfig->Read("/Processing/ThreadsPriority", 95l);

    _critSectionManager.Leave();
}

void asThreadsManager::OnClose(wxCloseEvent&) {
    if (GetRunningThreadsNb() > 0) {
        CleanArray();
    }
}

// Safety to manage by caller
int asThreadsManager::GetTotalThreadsNb() {
    return (int)_threads.size();
}

// Safe: Critical section defined within
int asThreadsManager::GetRunningThreadsNb(int type) {
    _critSectionManager.Enter();

    int counter = 0;

    for (auto& thread : _threads) {
        if (thread != nullptr) {
            if (thread->IsRunning()) {
                if (type == -1) {
                    counter++;
                } else {
                    if (thread->GetType() == type) {
                        counter++;
                    }
                }
            }
        }
    }

    _critSectionManager.Leave();

    return counter;
}

// Safe: Critical section defined within
int asThreadsManager::GetFreeDevice(int devicesNb) {
    if (devicesNb == 1) {
        return 0;
    }

    _critSectionManager.Enter();

    int deviceOccupancy = 0;
    int selectedDevice = 0;

    for (int device = 0; device < devicesNb; ++device) {
        int counter = 0;
        for (auto& thread : _threads) {
            if (thread != nullptr) {
                if (thread->IsRunning()) {
                    if (thread->GetDevice() == device) {
                        counter++;
                    }
                }
            }
        }
        if (counter == 0) {
            _critSectionManager.Leave();
            return device;
        }
        if (device == 0) {
            deviceOccupancy = counter;
        } else if (counter < deviceOccupancy) {
            selectedDevice = device;
        }
    }

    _critSectionManager.Leave();

    return selectedDevice;
}

// Safe: Critical section defined in GetRunningThreadsNb
int asThreadsManager::GetAvailableThreadsNb() {
    if (_maxThreadsNb < 1) Init();

    // Maximum threads nb
    int runningThreads = GetRunningThreadsNb();
    _critSectionManager.Enter();
    int nb = _maxThreadsNb - runningThreads;
    _critSectionManager.Leave();

    if (nb < 1) {
        nb = 1;
    }

    return nb;
}

// Safe: Critical section defined within
bool asThreadsManager::AddThread(asThread* thread) {
    // Check if needs to cleanup the threads array. Critical section locked within
    int runningThreads = GetRunningThreadsNb();
    if (runningThreads == 0) {
        CleanArray();
    }

    // Create
    if (thread->Create() != wxTHREAD_NO_ERROR) {
        wxLogError(_("Cannot create the thread !"));
        delete thread;
        return false;
    }

    // Set the thread Id
    wxASSERT(thread->GetId() >= 1);

    // Check the number of threads currently running
    if (GetAvailableThreadsNb() < 1) {
        wxLogWarning(_("The thread array is currently full."));
    }

    // Set priority
    if (_priority < 0) Init();
    thread->SetPriority(_priority);

    // Add to array
    _critSectionManager.Enter();
    _threads.push_back(thread);
    wxASSERT(thread->GetId() >= 1);
    _critSectionManager.Leave();

    // Run
    if (thread->Run() != wxTHREAD_NO_ERROR) {
        wxLogError(_("Can't run the thread!"));
        delete thread;
        return false;
    }

    _cancelled = false;
    _waitingUntilAllDone = true;

    return true;
}

void asThreadsManager::SetNull(wxThreadIdType id) {
    _critSectionManager.Enter();

    for (auto& thread : _threads) {
        if (thread != nullptr) {
            wxThreadIdType thisid = thread->GetId();

            if (thisid == id) {
                thread = nullptr;
                _critSectionManager.Leave();
                return;
            }
        }
    }

    wxLogError(_("Thread %d couldn't be removed."), id);

    _critSectionManager.Leave();
}

// Safe: Critical section defined within
bool asThreadsManager::CleanArray() {
    _critSectionManager.Enter();

    if (GetTotalThreadsNb() > 0) {
        for (auto& thread : _threads) {
            if (thread != nullptr) {
                _critSectionManager.Leave();
                return true;
            }
        }

        // If nothing is running, clear array.
        _threads.clear();
        _idCounter = 0;
    }

    _critSectionManager.Leave();

    return true;
}

void asThreadsManager::Wait(int type) {
    while (GetRunningThreadsNb(type) > 0) {
        wxMilliSleep(10);
    }
}

bool asThreadsManager::HasFreeThread(int type) {
    if (_maxThreadsNb < 1) Init();

    return _maxThreadsNb - GetRunningThreadsNb(type) > 0;
}

void asThreadsManager::WaitForFreeThread(int type) {
    if (_maxThreadsNb < 1) Init();

    while (_maxThreadsNb - GetRunningThreadsNb(type) <= 0) {
        wxMilliSleep(10);
    }
}

void asThreadsManager::PauseAll() {
    for (int iThread = 0; iThread < GetTotalThreadsNb(); iThread++) {
        if (_threads[iThread] != nullptr) {
            if (_threads[iThread]->IsRunning()) {
                //                _threads[iThread]->Pause();
            }
        }
    }
}

void asThreadsManager::ResumeAll() {
    for (int iThread = 0; iThread < GetTotalThreadsNb(); iThread++) {
        if (_threads[iThread] != nullptr) {
            if (_threads[iThread]->IsPaused()) {
                _threads[iThread]->Resume();
            }
        }
    }
}
