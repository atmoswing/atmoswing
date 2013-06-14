/** 
 *
 *  This file is part of the AtmoSwing software.
 *
 *  Copyright (c) 2008-2012  University of Lausanne, Pascal Horton (pascal.horton@unil.ch). 
 *  All rights reserved.
 *
 *  THIS CODE, SOFTWARE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT WARRANTY  
 *  OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR FITNESS FOR A PARTICULAR
 *  PURPOSE.
 *
 */
 
#ifndef ASTHREADSMANAGER_H
#define ASTHREADSMANAGER_H

#include <asIncludes.h>

// Predefinition
class asThread;

class asThreadsManager: public wxObject
{
public:
    /** Default constructor */
    asThreadsManager();

    /** Default destructor */
    virtual ~asThreadsManager();

    void Init();

    void OnClose(wxCloseEvent&);

    /** Add a thread
     * \param thread New thread
     */
    bool AddThread(asThread* thread);


    void Wait(int type);
    void WaitForFreeThread(int type);
    void SetNull(int id);
    bool CleanArray();
    void PauseAll();
    void ResumeAll();


    /** Count the total number of threads
     * \return The total number of threads
     */
    int GetTotalThreadsNb();

    /** Count the number of running threads
     * \return The number of running threads
     */
    int GetRunningThreadsNb(int type = -1);

    /** Count the number of available threads
     * \return The number of available threads
     */
    int GetAvailableThreadsNb();

    /** Access m_Cancelled
     * \return The current value of m_Cancelled
     */
    bool Cancelled()
    {
        wxCriticalSectionLocker lock(m_CritSectionManager);
        return m_Cancelled;
    }

    /** Set m_Cancelled to true
     */
    void Cancel()
    {
        wxCriticalSectionLocker lock(m_CritSectionManager);
        m_Cancelled = true;
    }

    /** Get a reference to the critical section of the data access
     * \return A reference to the critical section of the data access
     */
    wxCriticalSection& CritSectionNetCDF()
    {
        return m_CritSectionNetCDF;
    }

    /** Get a reference to the critical section of the config pointer
     * \return A reference to the critical section of the config pointer
     */
    wxCriticalSection& CritSectionConfig()
    {
        return m_CritSectionConfig;
    }

    wxCriticalSection& CritSectionTiCPP()
    {
        return m_CritSectionTiCPP;
    }

    wxCriticalSection& CritSectionPreloadedData()
    {
        return m_CritSectionPreloadedData;
    }

    /** Get a reference to the semaphore
     * \return A reference to the semaphore
     */
    wxSemaphore& SemAllDone()
    {
        return m_SemAllDone;
    }

    /** Get the m_WaitingUntilAllDone tag
     * \return The m_WaitingUntilAllDone current value
     */
    bool GetWaitingUntilAllDone()
    {
        return m_WaitingUntilAllDone;
    }

    /** Set the m_WaitingUntilAllDone tag
     * \param The new value
     */
    void SetWaitingUntilAllDone(bool val)
    {
        m_WaitingUntilAllDone = val;
    }


protected:
private:
    int m_IdCounter;
    std::vector < asThread* > m_Threads; //!< Member variable "m_Threads". All the threads currently alive (as soon as the thread terminates, it's removed from the array)
    wxCriticalSection m_CritSectionManager; //!< Member variable "m_CritSectionManager". Critical section.
    wxCriticalSection m_CritSectionPreloadedData;
    wxCriticalSection m_CritSectionNetCDF;
    wxCriticalSection m_CritSectionConfig;
    wxCriticalSection m_CritSectionTiCPP;
    wxSemaphore m_SemAllDone; //!< Member variable "m_SemAllDone". Semaphore used to wait for the threads to exit.
    bool m_WaitingUntilAllDone; //!< Member variable "m_WaitingUntilAllDone". The last exiting thread should post to m_semAllDone if this is true.
    bool m_Cancelled;
    int m_MaxThreadsNb;
    long m_Priority;
};

#endif // ASTHREADSMANAGER_H
