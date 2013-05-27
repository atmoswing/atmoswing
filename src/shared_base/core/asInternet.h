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
 
#ifndef ASINTERNET_H
#define ASINTERNET_H

#include <asIncludes.h>
#include <curl/curl.h>


class asInternet
{
public:
    asInternet();
    virtual ~asInternet();

    static void Init();
    static void Cleanup();

    int Download(const VectorString &urls, const VectorString &fileNames, const wxString &destinationDir);


protected:

private:
    friend class asThreadInternetDownload;
    /** File structure for cURL */
    struct HttpFile {
        const char *filename;
        FILE *stream;
    };
    static size_t WriteFile(void *buffer, size_t size, size_t nmemb, void *stream);

};

#endif // ASINTERNET_H
