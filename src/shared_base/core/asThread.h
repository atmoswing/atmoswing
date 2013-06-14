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
        PreprocessorGradients,
		MethodOptimizerRandomSet,
		MethodOptimizerGeneticAlgorithms
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
