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

#ifndef AS_THREAD_INTERNET_DOWNLOAD_H
#define AS_THREAD_INTERNET_DOWNLOAD_H

#include <asIncludes.h>
#include <asInternet.h>
#include <asThread.h>

class asThreadInternetDownload : public asThread {
   public:
    asThreadInternetDownload(const vwxs &urls, const vwxs &fileNames, const wxString &destinationDir, bool usesProxy,
                             const wxString &proxyAddress, long proxyPort, const wxString &proxyUser,
                             const wxString &proxyPasswd, int start, int end);

    ~asThreadInternetDownload() override = default;

    ExitCode Entry() override;

   protected:
   private:
    vwxs m_urls;
    vwxs m_fileNames;
    wxString m_destinationDir;
    bool m_usesProxy;
    wxString m_proxyAddress;
    long m_proxyPort;
    wxString m_proxyUser;
    wxString m_proxyPasswd;
    int m_start;
    int m_end;
};

#endif
