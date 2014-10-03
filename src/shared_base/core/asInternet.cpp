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
 
#include "asInternet.h"

#include <asThreadInternetDownload.h>
#include <asThreadsManager.h>

asInternet::asInternet()
{
    //ctor
}

asInternet::~asInternet()
{
    //dtor
}

void asInternet::Init()
{
    // Init cURL
    curl_global_init(CURL_GLOBAL_ALL);
}

void asInternet::Cleanup()
{
    // Cleanup cURL
    curl_global_cleanup();
}

size_t asInternet::WriteFile(void *buffer, size_t size, size_t nmemb, void *stream)
{
    struct HttpFile *out = (struct HttpFile *)stream;
    if(!out->stream) {
        // Open file for writing
        out->stream = fopen(out->filename, "wb");
        if(!out->stream)
            return 1; // failure, can't open file to write
    }
    return fwrite(buffer, size, nmemb, out->stream);
}

int asInternet::Download(const VectorString &urls, const VectorString &fileNames, const wxString &destinationDir)
{
    // Proxy
    bool usesProxy;
    wxConfigBase *pConfig = wxFileConfig::Get();
    pConfig->Read("/Internet/UsesProxy", &usesProxy, false);
    wxString proxyAddress = pConfig->Read("/Internet/ProxyAddress", wxEmptyString);
    long proxyPort;
    pConfig->Read("/Internet/ProxyPort", &proxyPort);
    wxString proxyUser = pConfig->Read("/Internet/ProxyUser", wxEmptyString);
    wxString proxyPasswd = pConfig->Read("/Internet/ProxyPasswd", wxEmptyString);

    // Get the number of connections
    //int threadsNb = wxMin(ThreadsManager().GetAvailableThreadsNb(), (int)fileNames.size());
    long parallelRequests = 5;
    pConfig->Read("/Internet/ParallelRequestsNb", &parallelRequests, 5l);

    if(parallelRequests>1)
    {
        // Disable message box
        g_pLog->DisableMessageBoxOnError();

        // Must initialize libcurl before any threads are started
        curl_global_init(CURL_GLOBAL_ALL);

        // Create threads
        int end = -1;
        parallelRequests = wxMin(parallelRequests, (int)fileNames.size());
        int threadType = -1;
        for (int i_threads=0; i_threads<parallelRequests; i_threads++)
        {
            int start = end+1;
            end = ceil(((float)(i_threads+1)*(float)(fileNames.size()-1)/(float)parallelRequests));
            wxASSERT(fileNames.size()>0);
            wxASSERT(end>=start);
            wxASSERT_MSG((unsigned)end<fileNames.size(), wxString::Format("Size of fileNames = %d, desired end = %d", (int)fileNames.size(), end));

            asThreadInternetDownload* thread = new asThreadInternetDownload(urls, fileNames, destinationDir, usesProxy, proxyAddress, proxyPort, proxyUser, proxyPasswd, start, end);
            threadType = thread->GetType();
            ThreadsManager().AddThread(thread);
        }

        // Wait until all done
        ThreadsManager().Wait(threadType);

        // Enable message box and flush the logs
        g_pLog->EnableMessageBoxOnError();
        g_pLog->Flush();

        // Check the files
        for (unsigned int i_file=0; i_file<fileNames.size(); i_file++)
        {
            wxString fileName = fileNames[i_file];
            wxString filePath = destinationDir + DS + fileName;
            if(!wxFileName::FileExists(filePath))
            {
                return asFAILED;
            }
        }
    }
    else
    {
        // Initialize
        CURL *curl;
        CURLcode res;
        curl = curl_easy_init();

        // Do the job
        if(curl) {
            #if wxUSE_GUI
                // The progress bar
                wxString dialogmessage = _("Downloading predictors.\n");
                asDialogProgressBar ProgressBar(dialogmessage, urls.size());
            #endif

            // Set a buffer for the error messages
            char* errorbuffer = new char[CURL_ERROR_SIZE];
            curl_easy_setopt(curl, CURLOPT_ERRORBUFFER,  errorbuffer);
            // Some servers don't like requests that are made without a user-agent field, so we provide one
            curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
            // Fail if the HTTP code returned is equal to or larger than 400
            curl_easy_setopt(curl, CURLOPT_FAILONERROR, true);
            // Maximum time in seconds that we allow the connection to the server to take. This only limits the connection phase.
            curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5);
            // Set a timeout period (in seconds) on the amount of time that the server is allowed to take in order to generate a response message for a command before the session is considered hung.
            curl_easy_setopt(curl, CURLOPT_FTP_RESPONSE_TIMEOUT, 10);

            for (unsigned int i_file=0; i_file<urls.size(); i_file++)
            {
                wxString fileName = fileNames[i_file];
                wxString filePath = destinationDir + DS + fileName;
                wxString url = urls[i_file];
                asLogMessage(wxString::Format(_("Downloading file %s."), filePath.c_str())); // Do not log the URL, it bugs !

                // Use of a wxFileName object to create the directory.
                wxFileName currentFilePath = wxFileName(filePath);
                if (!currentFilePath.DirExists())
                {
                    if (!currentFilePath.Mkdir(0777, wxPATH_MKDIR_FULL ))
                    {
                        asLogError(_("The directory to save real-time predictors data cannot be created."));
                        wxDELETE(errorbuffer);
                        return asFAILED;
                    }
                }

                #if wxUSE_GUI
                    // Update the progress bar
                    wxString updatedialogmessage = wxString::Format(_("Downloading file %s\n"), fileName.c_str()) + wxString::Format(_("Downloading: %d / %d files"), i_file+1, (int)urls.size());
                    if(!ProgressBar.Update(i_file, updatedialogmessage))
                    {
                        asLogMessage(_("The download has been canceled by the user."));
                        wxDELETE(errorbuffer);
                        return asCANCELLED;
                    }
                #endif

                // Download only if not already done
                if(!wxFileName::FileExists(filePath))
                {
                    // Instantiate the file structure
                    struct HttpFile file={
                        filePath.mb_str(), // Name to store the file as if succesful
                        NULL
                    };

                    // Define the URL
                    wxCharBuffer buffer=url.ToUTF8();
                    curl_easy_setopt(curl, CURLOPT_URL, buffer.data());
                    // Define our callback to get called when there's data to be written
                    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteFile);
                    // Set a pointer to our struct to pass to the callback
                    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &file);

                    // If a proxy is used
                    if (usesProxy)
                    {
                        if (!proxyAddress.IsEmpty())
                        {
                            wxCharBuffer proxyAddressBuffer = proxyAddress.ToUTF8();
                            curl_easy_setopt(curl, CURLOPT_PROXY, proxyAddressBuffer.data());
                        }
                        if (proxyPort>0)
                        {
                            curl_easy_setopt(curl, CURLOPT_PROXYPORT, proxyPort);
                        }
                        if (!proxyUser.IsEmpty())
                        {
                            wxString proxyLogin = proxyUser + ":" + proxyPasswd;
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
                        asLogWarning(wxString::Format(_("Failed downloading file. Curl error code: %d"), int(res)));
                        asLogWarning(wxString::Format(_("Curl error message: %s"), errorbuffer));
                        wxDELETE(errorbuffer);
                        return asFAILED;
                    }
                    else
                    {
                        asLogMessage(wxString::Format(_("File %d/%d downloaded successfully."), i_file+1, (int)urls.size()));
                    }
                }
            }

            #if wxUSE_GUI
                ProgressBar.Destroy();
            #endif

            // Always cleanup
            curl_easy_cleanup(curl);
            wxDELETE(errorbuffer);
        }
    }

    return asSUCCESS;
}
