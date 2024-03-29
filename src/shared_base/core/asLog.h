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

#ifndef AS_LOG_H
#define AS_LOG_H

#include "asIncludes.h"
#include "wx/log.h"

class wxFFile;

class asLog : public wxObject {
  public:
    enum LogTarget {
        File,
        Screen,
        Both
    };

    asLog();

    ~asLog() override;

    void ClearCurrentTarget();

    void CreateFile(const wxString& fileName);

    void CreateFileAtPath(const wxString& fullPath);

    void CreateFileOnly(const wxString& fileName);

    void CreateFileOnlyAtPath(const wxString& fullPath);

    void SetLevel(int val);

    static void PrintToConsole(const wxString& msg);

  protected:
  private:
    wxFFile* m_logFile;
    wxLogChain* m_logChain;
};

#if USE_GUI
class asLogGui : public wxLogGui {
  protected:
    void DoLogRecord(wxLogLevel level, const wxString& msg, const wxLogRecordInfo& info) override;
};
#endif

extern asLog* g_pLog;

asLog* Log();

void DeleteLog();

#endif
