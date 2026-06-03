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

#include <wx/filename.h>

#include "asIncludes.h"

asThreadInternetDownload::asThreadInternetDownload(const vwxs& urls, const vwxs& fileNames,
                                                   const wxString& destinationDir, bool usesProxy,
                                                   const wxString& proxyAddress, const long proxyPort,
                                                   const wxString& proxyUser, const wxString& proxyPasswd, int start,
                                                   int end)
    : asThread(),
      _urls(urls),
      _fileNames(fileNames),
      _destinationDir(destinationDir),
      _usesProxy(usesProxy),
      _proxyAddress(proxyAddress),
      _proxyPort(proxyPort),
      _proxyUser(proxyUser),
      _proxyPasswd(proxyPasswd),
      _start(start),
      _end((std::min)(end, static_cast<int>(fileNames.size()) - 1)) {
    wxASSERT(_end < urls.size());
    wxASSERT(_end < fileNames.size());
}

wxThread::ExitCode asThreadInternetDownload::Entry() {
    // Initialize
    CURL* curl;
    CURLcode res;
    curl = curl_easy_init();

    // Do the job
    if (curl) {
        // Set a buffer for the error messages
        auto errorBuffer = new char[CURL_ERROR_SIZE];
        curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, errorBuffer);
        // Some servers don't like requests that are made without a user-agent field, so we provide one
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "libcurl-agent/1.0");
        // Fail if the HTTP code returned is equal to or larger than 400
        curl_easy_setopt(curl, CURLOPT_FAILONERROR, true);
        // Maximum time in seconds that we allow the connection to the server to take. This only limits the connection
        // phase.
        curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 20);

        for (int iFile = _start; iFile <= _end; iFile++) {
            wxString fileName = _fileNames[iFile];
            wxString filePath = _destinationDir + DS + fileName;
            wxString url = _urls[iFile];
            wxLogVerbose(_("Downloading file %s."), filePath);  // Do not log the URL, it bugs !

            // Use of a wxFileName object to create the directory.
            wxFileName currentFilePath = wxFileName(filePath);
            if (!currentFilePath.Exists()) {
                if (!currentFilePath.Mkdir(0777, wxPATH_MKDIR_FULL)) {
                    wxLogError(_("The directory to save real-time predictors data cannot be created."));
                    curl_easy_cleanup(curl);
                    wxDELETEA(errorBuffer);
                    return (wxThread::ExitCode)-1;
                }
            }

            // Download only if not already done
            if (!wxFileName::FileExists(filePath)) {
                // Instantiate the file structure
                struct asInternet::HttpFile file = {filePath.mb_str(),  // Name to store the file as if succesful
                                                    nullptr};

                // Define the URL
                curl_easy_setopt(curl, CURLOPT_URL, static_cast<const char*>(url.mb_str(wxConvUTF8)));
                // Define our callback to get called when there's data to be written
                curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, asInternet::WriteFile);
                // Set a pointer to our struct to pass to the callback
                curl_easy_setopt(curl, CURLOPT_WRITEDATA, &file);
#if defined(__WIN32__)
                // Disable certificate check (CURLOPT_CAPATH does not work on Windows)
                curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, false);
#endif
                // If a proxy is used
                if (_usesProxy) {
                    if (!_proxyAddress.IsEmpty()) {
                        curl_easy_setopt(curl, CURLOPT_PROXY,
                                         static_cast<const char*>(_proxyAddress.mb_str(wxConvUTF8)));
                    }
                    if (_proxyPort > 0) {
                        curl_easy_setopt(curl, CURLOPT_PROXYPORT, _proxyPort);
                    }
                    if (!_proxyUser.IsEmpty()) {
                        wxString proxyLogin = _proxyUser + ":" + _proxyPasswd;
                        curl_easy_setopt(curl, CURLOPT_PROXYUSERPWD,
                                         static_cast<const char*>(proxyLogin.mb_str(wxConvUTF8)));
                    }
                }

                // Proceed
                res = curl_easy_perform(curl);

                // Close the local file
                if (file.stream) fclose(file.stream);

                // Log in case of failure
                if (CURLE_OK != res) {
                    wxLogError(_("Failed downloading file. Curl error code: %d"), int(res));
                    wxLogError(_("Curl error message: %s"), errorBuffer);
                    wxLogError(_("URL: %s"), url);
                    curl_easy_cleanup(curl);
                    wxDELETEA(errorBuffer);
                    return (wxThread::ExitCode)-1;
                } else {
                    wxLogVerbose(_("File %s downloaded successfully."), fileName);
                }
            }
        }

        // Always cleanup
        curl_easy_cleanup(curl);
        wxDELETEA(errorBuffer);
    }

    return (wxThread::ExitCode)0;
}
