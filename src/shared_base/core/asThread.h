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
 
#ifndef ASTHREAD_H
#define ASTHREAD_H

#include <wx/thread.h>

#include <asIncludes.h>

class asThread: public wxThread
{
public:

    //!< The thread status
    enum Status
    {
        Creating,
        Initializing,
        Waiting,
        Working,
        Done,
        Exiting,
        Canceling,
        Canceled,
        Pause,
        Error
    };

    //!< Possible thread types
    enum Type
    {
        ProcessorGetAnalogsDates,
        ProcessorGetAnalogsSubDates,
        PreprocessorGradients
    };

    /** Default constructor */
    asThread();

    /** Default destructor */
    virtual ~asThread();

    virtual ExitCode Entry();

    virtual void OnExit();

    int GetId()
    {
        return m_Id;
    }

    void SetId(int val)
    {
        m_Id = val;
    }

    asThread::Status GetStatus()
    {
        return m_Status;
    }

    void SetStatus(asThread::Status val)
    {
        m_Status = val;
    }

    asThread::Type GetType()
    {
        return m_Type;
    }

    bool IsDone()
    {
        return m_Status==Done;
    }

    bool IsRunning()
    {
        return m_Status==Working;
    }

    bool IsEnding()
    {
        return m_Status==Exiting;
    }

protected:
    asThread::Status m_Status;
    asThread::Type m_Type;

private:
    int m_Id;

};

#endif // ASTHREAD_H
