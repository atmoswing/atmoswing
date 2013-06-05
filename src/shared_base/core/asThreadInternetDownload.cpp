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
 
#include "asThreadInternetDownload.h"

#include <asTimeArray.h>
#include <asThreadsManager.h>


asThreadInternetDownload::asThreadInternetDownload(const VectorString &urls, const VectorString &fileNames, const wxString &destinationDir, bool usesProxy, const wxString &proxyAddress, const long proxyPort, const wxString &proxyUser, const wxString &proxyPasswd, int start, int end)
:
asThread()
{
    m_Status = Initializing;

    m_Urls = urls;
    m_FileNames = fileNames;
    m_DestinationDir = destinationDir;
    m_UsesProxy = usesProxy;
    m_ProxyAddress = proxyAddress;
    m_ProxyPort = proxyPort;
    m_ProxyUser = proxyUser;
    m_ProxyPasswd = proxyPasswd;
    m_Start = start;
    m_End = wxMin(end, (int)m_FileNames.size()-1);

    wxASSERT((unsigned)m_End<urls.size());
    wxASSERT((unsigned)m_End<fileNames.size());

    m_Status = Waiting;
}

asThreadInternetDownload::~asThreadInternetDownload()
{

}

wxThread::ExitCode asThreadInternetDownload::Entry()
{
    m_Status = Working;

    // Initialize
    CURL *curl;
    CURLcode res;
    curl = curl_easy_init();

    // Do the job
    if(curl) {
        // Set a buffer for the error messages
        char* errorbuffer = new char[CURL_ERROR_SIZE];
        curl_easy_setopt(curl, CURLOPT_ERRORBUFFER,  errorbuffer);
        // Some servers don't like requests that are made without a user-agent field, so we provide one
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
        // Verbose mode in case of error
        curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
        // Fail if the HTTP code returned is equal to or larger than 400
        curl_easy_setopt(curl, CURLOPT_FAILONERROR, true);
        // Maximum time in seconds that we allow the connection to the server to take. This only limits the connection phase.
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 20);

        for (int i_file=m_Start; i_file<=m_End; i_file++)
        {
            wxString fileName = m_FileNames[i_file];
            wxString filePath = m_DestinationDir + DS + fileName;
            wxString url = m_Urls[i_file];
            //asLogMessage(wxString::Format(_("Downloading file %s from url: %s"), filePath.c_str(), url.c_str())); Causes bug in string formatting
            asLogMessage(wxString::Format(_("Downloading file %s."), filePath.c_str()));

            // Use of a wxFileName object to create the directory.
            wxFileName currentFilePath = wxFileName(filePath);
            if (!currentFilePath.Mkdir(0777, wxPATH_MKDIR_FULL ))
            {
                asLogError(_("The directory to save real-time predictors data cannot be created."));
                return 0;
            }

            // Download only if not already done
            if(!wxFileName::FileExists(filePath))
            {
                // Instantiate the file structure
                struct asInternet::HttpFile file={
                    filePath.mb_str(), // Name to store the file as if succesful
                    NULL
                };

                // Define the URL
                wxCharBuffer buffer=url.ToUTF8();
                curl_easy_setopt(curl, CURLOPT_URL, buffer.data());
                // Define our callback to get called when there's data to be written
                curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, asInternet::WriteFile);
                // Set a pointer to our struct to pass to the callback
                curl_easy_setopt(curl, CURLOPT_WRITEDATA, &file);

                // If a proxy is used
                if (m_UsesProxy)
                {
                    if (!m_ProxyAddress.IsEmpty())
                    {
                        wxCharBuffer proxyAddressBuffer = m_ProxyAddress.ToUTF8();
                        curl_easy_setopt(curl, CURLOPT_PROXY, proxyAddressBuffer.data());
                    }
                    if (m_ProxyPort>0)
                    {
                        curl_easy_setopt(curl, CURLOPT_PROXYPORT, m_ProxyPort);
                    }
                    if (!m_ProxyUser.IsEmpty())
                    {
                        wxString proxyLogin = m_ProxyUser + ":" + m_ProxyPasswd;
                        wxCharBuffer proxyLoginBuffer = proxyLogin.ToUTF8();
                        curl_easy_setopt(curl, CURLOPT_PROXYUSERPWD, proxyLoginBuffer.data());
                    }
                }

                // Proceed
                res = curl_easy_perform(curl);

                // Close the local file
                if(file.stream) fclose(file.stream);

                // Log in case of failure
                if(CURLE_OK != res) {
                    asLogError(wxString::Format(_("Failed downloading file. Curl error code: %d"), res));
                    asLogError(wxString::Format(_("Curl error message: %s"), errorbuffer));
                    return 0;
                }
                else
                {
                    asLogMessage(wxString::Format(_("File %s downloaded successfully."), fileName.c_str()));
                }
            }
        }

        // Always cleanup
        curl_easy_cleanup(curl);
        wxDELETE(errorbuffer);
    }

    m_Status = Done;

    return 0;
}
