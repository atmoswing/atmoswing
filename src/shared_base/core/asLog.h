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

    virtual void Flush();

    bool IsVerbose();

    void Error(const wxString &msg);

    void Warning(const wxString &msg);

    void Message(const wxString &msg, bool force=false);

    void State(const wxString &msg);

    bool IsMessageBoxOnErrorEnabled()
    {
        return m_messageBoxOnError;
    }

    void DisableMessageBoxOnError()
    {
        m_messageBoxOnError = false;
    }

    void EnableMessageBoxOnError()
    {
        m_messageBoxOnError = true;
    }

    void StopLogging()
    {
        m_active = false;
    }

    void ResumeLogging()
    {
        m_active = true;
    }

    void SetLevel(int val)
    {
        m_level = val;
    }

    void SetTarget(int val)
    {
        m_target = val;
    }

    wxString GetState()
    {
        m_critSectionLog.Enter();
        wxString state = m_state;
        m_critSectionLog.Leave();
        return state;
    }

protected:
private:
    wxFFile *m_logFile; //!< Member variable "m_logFile".
    wxLogChain *m_logChain; //!< Member variable "m_logChain".
    wxCriticalSection m_critSectionLog; //!< Member variable "m_critSectionLog". Critical section.
    int m_level; //!< Member variable "m_level". 1: only errors, 2: errors & warnings, 3: all logs.
    int m_target;
    bool m_active;
    bool m_messageBoxOnError;
    wxString m_state;
};

#endif // ASLOG_H
