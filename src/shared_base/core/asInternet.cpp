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

#include "asInternet.h"

#include "asThreadInternetDownload.h"

void asInternet::Init() {
    // Init cURL
    curl_global_init(CURL_GLOBAL_ALL);
}

void asInternet::Cleanup() {
    // Cleanup cURL
    curl_global_cleanup();
}

size_t asInternet::WriteFile(void* buffer, size_t size, size_t nmemb, void* stream) {
    auto* out = (struct HttpFile*)stream;
    if (!out->stream) {
        // Open file for writing
        out->stream = fopen(out->fileName, "wb");
        if (!out->stream) return 1;  // failure, can't open file to write
    }
    return fwrite(buffer, size, nmemb, out->stream);
}

int asInternet::Download(const vwxs& urls, const vwxs& fileNames, const wxString& destinationDir) {
    // Proxy
    wxConfigBase* pConfig = wxFileConfig::Get();
    bool usesProxy = pConfig->ReadBool("/Internet/UsesProxy", false);
    wxString proxyAddress = pConfig->Read("/Internet/ProxyAddress", wxEmptyString);
    long proxyPort = pConfig->ReadLong("/Internet/ProxyPort", 8080);
    wxString proxyUser = pConfig->Read("/Internet/ProxyUser", wxEmptyString);
    wxString proxyPasswd = pConfig->Read("/Internet/ProxyPasswd", wxEmptyString);

    // Get the number of connections
    // int threadsNb = wxMin(ThreadsManager().GetAvailableThreadsNb(), (int)fileNames.size());
    long parallelRequests = pConfig->ReadLong("/Internet/ParallelRequestsNb", 5l);

    if (parallelRequests > 1) {
        // Must initialize libcurl before any threads are started
        curl_global_init(CURL_GLOBAL_ALL);

        // Create threads
        int end = -1;
        parallelRequests = wxMin(parallelRequests, (int)fileNames.size());
        int threadType = -1;
        for (int iThread = 0; iThread < parallelRequests; iThread++) {
            int nbFiles = fileNames.size();
            if (end >= nbFiles - 1) {
                break;
            }
            int start = end + 1;
            end = wxMin(start + ceil((float)fileNames.size() / (float)parallelRequests) - 1, (int)fileNames.size() - 1);
            wxASSERT(!fileNames.empty());
            wxASSERT(end >= start);
            wxASSERT_MSG(end < fileNames.size(),
                         asStrF("Size of fileNames = %d, desired end = %d", (int)fileNames.size(), end));

            auto* thread = new asThreadInternetDownload(urls, fileNames, destinationDir, usesProxy, proxyAddress,
                                                        proxyPort, proxyUser, proxyPasswd, start, end);
            threadType = thread->GetType();
            ThreadsManager().AddThread(thread);
        }

        // Wait until all done
        ThreadsManager().Wait(threadType);

        // Enable message box and flush the logs
#if USE_GUI
        wxTheApp->Yield();
#else
        wxLog::FlushActive();
        wxTheApp->Yield();
#endif

        // Check the files
        for (const auto& fileName : fileNames) {
            wxString filePath = destinationDir + DS + fileName;
            if (!wxFileName::FileExists(filePath)) {
                return asFAILED;
            }
        }
    } else {
        // Initialize
        CURL* curl;
        CURLcode res;
        curl = curl_easy_init();

        // Do the job
        if (curl) {
#if USE_GUI
            // The progress bar
            wxString msg = _("Downloading predictors.\n");
            asDialogProgressBar progressBar(msg, urls.size());
#endif

            // Set a buffer for the error messages
            auto* errorBfr = new char[CURL_ERROR_SIZE];
            curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, errorBfr);
            // Some servers don't like requests that are made without a user-agent field, so we provide one
            curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
            // Fail if the HTTP code returned is equal to or larger than 400
            curl_easy_setopt(curl, CURLOPT_FAILONERROR, true);
            // Maximum time in seconds that we allow the connection to the server to take. This only limits the
            // connection phase.
            curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5);
            // Set a timeout period (in seconds) on the amount of time that the server is allowed to take in order to
            // generate a response message for a command before the session is considered hung.
            curl_easy_setopt(curl, CURLOPT_FTP_RESPONSE_TIMEOUT, 10);

            for (int iFile = 0; iFile < urls.size(); iFile++) {
                wxString fileName = fileNames[iFile];
                wxString filePath = destinationDir + DS + fileName;
                wxString url = urls[iFile];
                wxLogVerbose(_("Downloading file %s."), filePath);  // Do not log the URL, it bugs !

                // Use of a wxFileName object to create the directory.
                wxFileName currentFilePath = wxFileName(filePath);
                if (!currentFilePath.Exists()) {
                    if (!currentFilePath.Mkdir(0777, wxPATH_MKDIR_FULL)) {
                        wxLogError(_("The directory to save real-time predictors data cannot be created."));
                        curl_easy_cleanup(curl);
                        wxDELETEA(errorBfr);
                        return asFAILED;
                    }
                }

#if USE_GUI
                // Update the progress bar
                wxString updateMsg = asStrF(_("Downloading file %s\n"), fileName) +
                                     asStrF(_("Downloading: %d / %d files"), iFile + 1, (int)urls.size());
                if (!progressBar.Update(iFile, updateMsg)) {
                    wxLogVerbose(_("The download has been canceled by the user."));
                    wxDELETEA(errorBfr);
                    return asCANCELLED;
                }
#endif

                // Download only if not already done
                if (!wxFileName::FileExists(filePath)) {
                    // Instantiate the file structure
                    struct HttpFile file = {filePath.mb_str(),  // Name to store the file as if successful
                                            nullptr};

                    // Define the URL
                    curl_easy_setopt(curl, CURLOPT_URL, (const char*)url.mb_str(wxConvUTF8));
                    // Define our callback to get called when there's data to be written
                    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteFile);
                    // Set a pointer to our struct to pass to the callback
                    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &file);
#if defined(__WIN32__)
                    // Disable certificate check (CURLOPT_CAPATH does not work on Windows)
                    curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, false);
#endif

                    // If a proxy is used
                    if (usesProxy) {
                        if (!proxyAddress.IsEmpty()) {
                            curl_easy_setopt(curl, CURLOPT_PROXY, (const char*)proxyAddress.mb_str(wxConvUTF8));
                        }
                        if (proxyPort > 0) {
                            curl_easy_setopt(curl, CURLOPT_PROXYPORT, proxyPort);
                        }
                        if (!proxyUser.IsEmpty()) {
                            wxString proxyLogin = proxyUser + ":" + proxyPasswd;
                            curl_easy_setopt(curl, CURLOPT_PROXYUSERPWD, (const char*)proxyLogin.mb_str(wxConvUTF8));
                        }
                    }

                    // Proceed
                    res = curl_easy_perform(curl);

                    // Close the local file
                    if (file.stream) fclose(file.stream);

                    // Log in case of failure
                    if (CURLE_OK != res) {
                        wxLogWarning(_("Failed downloading file. Curl error code: %d"), int(res));
                        wxLogWarning(_("Curl error message: %s"), errorBfr);
                        wxLogWarning(_("URL: %s"), url);
                        curl_easy_cleanup(curl);
                        wxDELETEA(errorBfr);
                        return asFAILED;
                    } else {
                        wxLogVerbose(_("File %d/%d downloaded successfully."), iFile + 1, (int)urls.size());
                    }
                }
            }

#if USE_GUI
            progressBar.Destroy();
#endif

            // Always cleanup
            curl_easy_cleanup(curl);
            wxDELETEA(errorBfr);
        }
    }

    return asSUCCESS;
}
