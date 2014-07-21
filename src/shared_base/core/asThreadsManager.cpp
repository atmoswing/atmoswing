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
 * The Original Software is AtmoSwing. The Initial Developer of the 
 * Original Software is Pascal Horton of the University of Lausanne. 
 * All Rights Reserved.
 * 
 */

/*
 * Portions Copyright 2008-2013 University of Lausanne.
 */
 
#include <asThreadsManager.h>
#include <asThread.h>

// Safe: Critical section defined within
asThreadsManager::asThreadsManager()
{
    m_CritSectionManager.Enter();

    m_Cancelled = false;
    m_IdCounter = 0;
    m_MaxThreadsNb = -1;
    m_Priority = -1;

    m_CritSectionManager.Leave();
}

asThreadsManager::~asThreadsManager()
{

}

void asThreadsManager::Init()
{
    m_CritSectionManager.Enter();

    wxConfigBase *pConfig = wxFileConfig::Get();
    m_MaxThreadsNb = wxThread::GetCPUCount();
    if (m_MaxThreadsNb==-1) m_MaxThreadsNb = 2;
    pConfig->Read("/Standard/ProcessingMaxThreadNb", &m_MaxThreadsNb, m_MaxThreadsNb);
    m_Priority = pConfig->Read("/Standard/ProcessingThreadsPriority", 95l);

    m_CritSectionManager.Leave();
}

void asThreadsManager::OnClose(wxCloseEvent&)
{
    // Check if we have any threads running first
    int count = GetRunningThreadsNb();

    // Delete
    if ( count )
    {
        CleanArray();
    }
}

// Safety to manage by caller
int asThreadsManager::GetTotalThreadsNb()
{
    int size = m_Threads.size();

    return size;
}

// Safe: Critical section defined within
int asThreadsManager::GetRunningThreadsNb(int type)
{
    m_CritSectionManager.Enter();

    int counter = 0;

    for (unsigned int i_threads=0; i_threads<m_Threads.size(); i_threads++)
    {
        if (m_Threads[i_threads]!=NULL)
        {
            if ( (m_Threads[i_threads]->GetStatus()!=asThread::Done) & (m_Threads[i_threads]->GetStatus()!=asThread::Exiting) & (m_Threads[i_threads]->GetStatus()!=asThread::Canceled) & (m_Threads[i_threads]->GetStatus()!=asThread::Error) )
            {
                if (type==-1)
                {
                    counter++;
                }
                else
                {
                    if (m_Threads[i_threads]->GetType()==type)
                    {
                        counter++;
                    }
                }
            }
        }
    }

    m_CritSectionManager.Leave();

    return counter;
}

// Safe: Critical section defined in GetRunningThreadsNb
int asThreadsManager::GetAvailableThreadsNb()
{
    if(m_MaxThreadsNb<1) Init();

    // Maximum threads nb
    int runningThreads = GetRunningThreadsNb();
    asLogMessage(wxString::Format(_("%d running threads (checking available threads)."), runningThreads));
    m_CritSectionManager.Enter();
    int nb = m_MaxThreadsNb-runningThreads;
    m_CritSectionManager.Leave();

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
    m_CritSectionManager.Enter();
    wxASSERT(m_IdCounter>=0);
    m_IdCounter++;
    thread->SetId(m_IdCounter);
    wxASSERT(thread->GetId()>=1);
    m_CritSectionManager.Leave();

    // Check the number of threads currently running
    if(GetAvailableThreadsNb()<1)
    {
        asLogWarning(_("The thread array is currently full."));
    }

    // Set priority
    if (m_Priority<0) Init();
    thread->SetPriority((int)m_Priority);

    // Add to array
    m_CritSectionManager.Enter();
    m_Threads.push_back(thread);
    wxASSERT(thread->GetId()>=1);
    asLogMessage(wxString::Format(_("A new thread was created (id=%d)."), (int)thread->GetId()));
    m_CritSectionManager.Leave();

    // Run
    if (thread->Run() != wxTHREAD_NO_ERROR )
    {
        wxLogError("Can't run the thread!");
        delete thread;
        return false;
    }

    m_Cancelled = false;
    m_WaitingUntilAllDone = true;

    return true;
}

void asThreadsManager::SetNull(int id)
{
    m_CritSectionManager.Enter();

    for (unsigned int i_threads=0; i_threads<m_Threads.size(); i_threads++)
    {
        if (m_Threads[i_threads]!=NULL)
        {
            int thisid = m_Threads[i_threads]->GetId();

            if (thisid==id)
            {
                m_Threads[i_threads]=NULL;
                m_CritSectionManager.Leave();
                return;
            }
        }
    }

    asLogError(wxString::Format(_("Thread %d couldn't be removed."), id));

    m_CritSectionManager.Leave();

}

// Safe: Critical section defined within
bool asThreadsManager::CleanArray()
{
    m_CritSectionManager.Enter();

    if(GetTotalThreadsNb()>0)
    {
        for (unsigned int i_threads=0; i_threads<m_Threads.size(); i_threads++)
        {
            if (m_Threads[i_threads]!=NULL)
            {
                m_CritSectionManager.Leave();
                return true;
            }
        }

        // If nothing is running, clear array.
        m_Threads.clear();
        m_IdCounter = 0;

        asLogMessage(_("Thread array cleared."));
    }

    m_CritSectionManager.Leave();

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

void asThreadsManager::WaitForFreeThread(int type)
{
    if(m_MaxThreadsNb<1) Init();

    while(m_MaxThreadsNb-GetRunningThreadsNb(type)<=0)
    {
        wxMilliSleep(10);
    }

    asLogMessage(_("A thread is available."));
}

void asThreadsManager::PauseAll()
{
    for (int i_threads=0; i_threads<GetTotalThreadsNb(); i_threads++)
    {
        if (m_Threads[i_threads]!=NULL)
        {
            if(m_Threads[i_threads]->IsRunning())
            {
//                m_Threads[i_threads]->Pause();
            }
        }
    }
}

void asThreadsManager::ResumeAll()
{
    for (int i_threads=0; i_threads<GetTotalThreadsNb(); i_threads++)
    {
        if (m_Threads[i_threads]!=NULL)
        {
            if(m_Threads[i_threads]->IsPaused())
            {
                m_Threads[i_threads]->Resume();
            }
        }
    }
}
