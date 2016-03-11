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

    /** Access m_cancelled
     * \return The current value of m_cancelled
     */
    bool Cancelled()
    {
        wxCriticalSectionLocker lock(m_critSectionManager);
        return m_cancelled;
    }

    /** Set m_cancelled to true
     */
    void Cancel()
    {
        wxCriticalSectionLocker lock(m_critSectionManager);
        m_cancelled = true;
    }

    /** Get a reference to the critical section of the data access
     * \return A reference to the critical section of the data access
     */
    wxCriticalSection& CritSectionNetCDF()
    {
        return m_critSectionNetCDF;
    }

    /** Get a reference to the critical section of the config pointer
     * \return A reference to the critical section of the config pointer
     */
    wxCriticalSection& CritSectionConfig()
    {
        return m_critSectionConfig;
    }

    wxCriticalSection& CritSectionPreloadedData()
    {
        return m_critSectionPreloadedData;
    }

    /** Get a reference to the semaphore
     * \return A reference to the semaphore
     */
    wxSemaphore& SemAllDone()
    {
        return m_semAllDone;
    }

    /** Get the m_waitingUntilAllDone tag
     * \return The m_waitingUntilAllDone current value
     */
    bool GetWaitingUntilAllDone()
    {
        return m_waitingUntilAllDone;
    }

    /** Set the m_waitingUntilAllDone tag
     * \param The new value
     */
    void SetWaitingUntilAllDone(bool val)
    {
        m_waitingUntilAllDone = val;
    }


protected:
private:
    int m_idCounter;
    std::vector < asThread* > m_threads; //!< Member variable "m_threads". All the threads currently alive (as soon as the thread terminates, it's removed from the array)
    wxCriticalSection m_critSectionManager; //!< Member variable "m_critSectionManager". Critical section.
    wxCriticalSection m_critSectionPreloadedData;
    wxCriticalSection m_critSectionNetCDF;
    wxCriticalSection m_critSectionConfig;
    wxSemaphore m_semAllDone; //!< Member variable "m_semAllDone". Semaphore used to wait for the threads to exit.
    bool m_waitingUntilAllDone; //!< Member variable "m_waitingUntilAllDone". The last exiting thread should post to m_semAllDone if this is true.
    bool m_cancelled;
    int m_maxThreadsNb;
    long m_priority;
};

#endif // ASTHREADSMANAGER_H
