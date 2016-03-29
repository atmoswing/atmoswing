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
 
#include <asThreadsManager.h>
#include <asThread.h>

// Safe: Critical section defined within
asThreadsManager::asThreadsManager()
{
    m_critSectionManager.Enter();

    m_cancelled = false;
    m_idCounter = 0;
    m_maxThreadsNb = -1;
    m_priority = -1;

    m_critSectionManager.Leave();
}

asThreadsManager::~asThreadsManager()
{
    if (GetRunningThreadsNb() > 0) {
        CleanArray();
    }
}

void asThreadsManager::Init()
{
    m_critSectionManager.Enter();

    wxConfigBase *pConfig = wxFileConfig::Get();
    m_maxThreadsNb = wxThread::GetCPUCount();
    if (m_maxThreadsNb==-1) m_maxThreadsNb = 2;
    pConfig->Read("/Processing/MaxThreadNb", &m_maxThreadsNb, m_maxThreadsNb);
    m_priority = pConfig->Read("/Processing/ThreadsPriority", 95l);

    m_critSectionManager.Leave();
}

void asThreadsManager::OnClose(wxCloseEvent&)
{
    if (GetRunningThreadsNb() > 0) {
        CleanArray();
    }
}

// Safety to manage by caller
int asThreadsManager::GetTotalThreadsNb()
{
    int size = m_threads.size();

    return size;
}

// Safe: Critical section defined within
int asThreadsManager::GetRunningThreadsNb(int type)
{
    m_critSectionManager.Enter();

    int counter = 0;

    for (unsigned int i_threads=0; i_threads<m_threads.size(); i_threads++)
    {
        if (m_threads[i_threads]!=NULL)
        {
            if ( (m_threads[i_threads]->GetStatus()!=asThread::Done) & (m_threads[i_threads]->GetStatus()!=asThread::Exiting) & (m_threads[i_threads]->GetStatus()!=asThread::Canceled) & (m_threads[i_threads]->GetStatus()!=asThread::Error) )
            {
                if (type==-1)
                {
                    counter++;
                }
                else
                {
                    if (m_threads[i_threads]->GetType()==type)
                    {
                        counter++;
                    }
                }
            }
        }
    }

    m_critSectionManager.Leave();

    return counter;
}

// Safe: Critical section defined in GetRunningThreadsNb
int asThreadsManager::GetAvailableThreadsNb()
{
    if(m_maxThreadsNb<1) Init();

    // Maximum threads nb
    int runningThreads = GetRunningThreadsNb();
    asLogMessage(wxString::Format(_("%d running threads (checking available threads)."), runningThreads));
    m_critSectionManager.Enter();
    int nb = m_maxThreadsNb-runningThreads;
    m_critSectionManager.Leave();

    if (nb<1)
    {
        nb = 1;
    }

    return nb;
}

// Safe: Critical section defined within
bool asThreadsManager::AddThread(asThread* thread)
{
    // Check if needs to cleanup the threads array. Critical section locked within
    int runningThreads = GetRunningThreadsNb();
    asLogMessage(wxString::Format(_("%d running threads before addition of a new thread."), runningThreads));
    if(runningThreads==0)
    {
        CleanArray();
    }

    // Create
    if (thread->Create()!=wxTHREAD_NO_ERROR)
    {
        asLogError(_("Cannot create the thread !"));
        delete thread;
        return false;
    }

    // Set the thread Id
    m_critSectionManager.Enter();
    wxASSERT(m_idCounter>=0);
    m_idCounter++;
    thread->SetId(m_idCounter);
    wxASSERT(thread->GetId()>=1);
    m_critSectionManager.Leave();

    // Check the number of threads currently running
    if(GetAvailableThreadsNb()<1)
    {
        asLogWarning(_("The thread array is currently full."));
    }

    // Set priority
    if (m_priority<0) Init();
    thread->SetPriority((int)m_priority);

    // Add to array
    m_critSectionManager.Enter();
    m_threads.push_back(thread);
    wxASSERT(thread->GetId()>=1);
    asLogMessage(wxString::Format(_("A new thread was created (id=%d)."), (int)thread->GetId()));
    m_critSectionManager.Leave();

    // Run
    if (thread->Run() != wxTHREAD_NO_ERROR )
    {
        wxLogError("Can't run the thread!");
        delete thread;
        return false;
    }

    m_cancelled = false;
    m_waitingUntilAllDone = true;

    return true;
}

void asThreadsManager::SetNull(int id)
{
    m_critSectionManager.Enter();

    for (unsigned int i_threads=0; i_threads<m_threads.size(); i_threads++)
    {
        if (m_threads[i_threads]!=NULL)
        {
            int thisid = m_threads[i_threads]->GetId();

            if (thisid==id)
            {
                m_threads[i_threads]=NULL;
                m_critSectionManager.Leave();
                return;
            }
        }
    }

    asLogError(wxString::Format(_("Thread %d couldn't be removed."), id));

    m_critSectionManager.Leave();

}

// Safe: Critical section defined within
bool asThreadsManager::CleanArray()
{
    m_critSectionManager.Enter();

    if(GetTotalThreadsNb()>0)
    {
        for (unsigned int i_threads=0; i_threads<m_threads.size(); i_threads++)
        {
            if (m_threads[i_threads]!=NULL)
            {
                m_critSectionManager.Leave();
                return true;
            }
        }

        // If nothing is running, clear array.
        m_threads.clear();
        m_idCounter = 0;

        asLogMessage(_("Thread array cleared."));
    }

    m_critSectionManager.Leave();

    return true;
}

void asThreadsManager::Wait(int type)
{
    while(GetRunningThreadsNb(type)>0)
    {
        wxMilliSleep(10);
    }

    asLogMessage(_("All threads have done."));
}

bool asThreadsManager::HasFreeThread(int type)
{
    if(m_maxThreadsNb<1) Init();

    return m_maxThreadsNb-GetRunningThreadsNb(type)>0;
}

void asThreadsManager::WaitForFreeThread(int type)
{
    if(m_maxThreadsNb<1) Init();

    while(m_maxThreadsNb-GetRunningThreadsNb(type)<=0)
    {
        wxMilliSleep(10);
    }

    asLogMessage(_("A thread is available."));
}

void asThreadsManager::PauseAll()
{
    for (int i_threads=0; i_threads<GetTotalThreadsNb(); i_threads++)
    {
        if (m_threads[i_threads]!=NULL)
        {
            if(m_threads[i_threads]->IsRunning())
            {
//                m_threads[i_threads]->Pause();
            }
        }
    }
}

void asThreadsManager::ResumeAll()
{
    for (int i_threads=0; i_threads<GetTotalThreadsNb(); i_threads++)
    {
        if (m_threads[i_threads]!=NULL)
        {
            if(m_threads[i_threads]->IsPaused())
            {
                m_threads[i_threads]->Resume();
            }
        }
    }
}
