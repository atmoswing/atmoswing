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
 
#ifndef asThreadInternetDownload_H
#define asThreadInternetDownload_H

#include <asThread.h>
#include <asIncludes.h>
#include <asInternet.h>

class asThreadInternetDownload: public asThread
{
public:
    /** Default constructor */
    asThreadInternetDownload(const VectorString &urls, const VectorString &fileNames, const wxString &destinationDir, bool usesProxy, const wxString &proxyAddress, const long proxyPort, const wxString &proxyUser, const wxString &proxyPasswd, int start, int end);
    /** Default destructor */
    virtual ~asThreadInternetDownload();

    virtual ExitCode Entry();


protected:
private:
    VectorString m_Urls;
    VectorString m_FileNames;
    wxString m_DestinationDir;
    bool m_UsesProxy;
    wxString m_ProxyAddress;
    long m_ProxyPort;
    wxString m_ProxyUser;
    wxString m_ProxyPasswd;
    int m_Start;
    int m_End;

};

#endif // asThreadInternetDownload_H
