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
