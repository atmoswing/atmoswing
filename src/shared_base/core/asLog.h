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
 
#ifndef ASLOG_H
#define ASLOG_H

#include "wx/log.h"

#include <asIncludes.h>

class wxFFile;

class asLog: public wxObject
{
public:
    enum LogTarget //!< Enumaration of log targets options
    {
        File,
        Screen,
        Both
    };

    /** Default constructor */
    asLog();

    /** Default destructor */
    virtual ~asLog();

    bool CreateFile(const wxString &fileName);

    bool CreateFileOnly(const wxString &fileName);

    bool CreateFileOnlyAtPath(const wxString &fullPath);

    void SetLogNull();

    static void Suspend();

    static void Resume();

    void Error(const wxString &msg);

    void Warning(const wxString &msg);

    void Message(const wxString &msg, bool force=false);

    void State(const wxString &msg);

    bool IsMessageBoxOnErrorEnabled()
    {
        return m_MessageBoxOnError;
    }

    void DisableMessageBoxOnError()
    {
        m_MessageBoxOnError = false;
    }

    void EnableMessageBoxOnError()
    {
        m_MessageBoxOnError = true;
    }

    void StopLogging()
    {
        m_Active = false;
    }

    void ResumeLogging()
    {
        m_Active = true;
    }

    void RemoveDuplicates()
    {
        m_RemoveDuplicates = true;
    }

    void AllowDuplicates()
    {
        m_RemoveDuplicates = false;
    }

    void SetLevel(int val)
    {
        m_Level = val;
    }

    void SetTarget(int val)
    {
        m_Target = val;
    }

    wxString GetState()
    {
        m_CritSectionLog.Enter();
        wxString state = m_State;
        m_CritSectionLog.Leave();
        return state;
    }

protected:
private:
    wxFFile *m_LogFile; //!< Member variable "m_LogFile".
    wxLogChain *m_LogChain; //!< Member variable "m_LogChain".
    wxCriticalSection m_CritSectionLog; //!< Member variable "m_CritSectionLog". Critical section.
    int m_Level; //!< Member variable "m_Level". 1: only errors, 2: errors & warnings, 3: all logs.
    int m_Target;
    bool m_Active;
    bool m_MessageBoxOnError;
    bool m_RemoveDuplicates;
    bool m_SignalDuplicates;
    wxString m_Buffer;
    wxString m_State;
};

#endif // ASLOG_H
