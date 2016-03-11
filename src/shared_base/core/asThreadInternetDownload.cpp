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
 
#include "asThreadInternetDownload.h"

#include <asTimeArray.h>
#include <asThreadsManager.h>


asThreadInternetDownload::asThreadInternetDownload(const VectorString &urls, const VectorString &fileNames, const wxString &destinationDir, bool usesProxy, const wxString &proxyAddress, const long proxyPort, const wxString &proxyUser, const wxString &proxyPasswd, int start, int end)
:
asThread()
{
    m_status = Initializing;

    m_urls = urls;
    m_fileNames = fileNames;
    m_destinationDir = destinationDir;
    m_usesProxy = usesProxy;
    m_proxyAddress = proxyAddress;
    m_proxyPort = proxyPort;
    m_proxyUser = proxyUser;
    m_proxyPasswd = proxyPasswd;
    m_start = start;
    m_End = wxMin(end, (int)m_fileNames.size()-1);

    wxASSERT((unsigned)m_End<urls.size());
    wxASSERT((unsigned)m_End<fileNames.size());

    m_status = Waiting;
}

asThreadInternetDownload::~asThreadInternetDownload()
{

}

wxThread::ExitCode asThreadInternetDownload::Entry()
{
    m_status = Working;

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
        // Fail if the HTTP code returned is equal to or larger than 400
        curl_easy_setopt(curl, CURLOPT_FAILONERROR, true);
        // Maximum time in seconds that we allow the connection to the server to take. This only limits the connection phase.
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 20);

        for (int i_file=m_start; i_file<=m_End; i_file++)
        {
            wxString fileName = m_fileNames[i_file];
            wxString filePath = m_destinationDir + DS + fileName;
            wxString url = m_urls[i_file];
            asLogMessage(wxString::Format(_("Downloading file %s."), filePath)); // Do not log the URL, it bugs !

            // Use of a wxFileName object to create the directory.
            wxFileName currentFilePath = wxFileName(filePath);
            if (!currentFilePath.DirExists())
            {
                if (!currentFilePath.Mkdir(0777, wxPATH_MKDIR_FULL ))
                {
                    asLogError(_("The directory to save real-time predictors data cannot be created."));
					wxDELETE(errorbuffer);
                    return 0;
                }
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
                if (m_usesProxy)
                {
                    if (!m_proxyAddress.IsEmpty())
                    {
                        wxCharBuffer proxyAddressBuffer = m_proxyAddress.ToUTF8();
                        curl_easy_setopt(curl, CURLOPT_PROXY, proxyAddressBuffer.data());
                    }
                    if (m_proxyPort>0)
                    {
                        curl_easy_setopt(curl, CURLOPT_PROXYPORT, m_proxyPort);
                    }
                    if (!m_proxyUser.IsEmpty())
                    {
                        wxString proxyLogin = m_proxyUser + ":" + m_proxyPasswd;
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
                    asLogError(_("Failed downloading file."));
                    asLogError(wxString::Format(_("Curl error message: %s"), errorbuffer));
					wxDELETE(errorbuffer);
                    return 0;
                }
                else
                {
                    asLogMessage(wxString::Format(_("File %s downloaded successfully."), fileName));
                }
            }
        }

        // Always cleanup
        curl_easy_cleanup(curl);
        wxDELETE(errorbuffer);
    }

    m_status = Done;

    return 0;
}
