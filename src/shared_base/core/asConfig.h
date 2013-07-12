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
 
#ifndef ASCONFIG_H
#define ASCONFIG_H

#include <asIncludes.h>
#include "wx/fileconf.h"        // wxFileConfig

class asConfig : public wxObject
{
    public:
        /** Default constructor */
        asConfig();
        /** Destructor */
        virtual ~asConfig();

        /** Get log directory
        * \return The full path
        */
        static wxString GetLogDir();

        /** Get temp directory
        * \return The full path
        */
        static wxString GetTempDir();

        /** Get temp directory
        * \param prefix Prefix to use for the temporary file name construction
        * \return The full path
        * \note From wxFileName::CreateTempFileName. Change to avoid file auto creation.
        */
        static wxString CreateTempFileName(const wxString& prefix);

        /** Get common data directory
        * \return The full path
        */
        static wxString GetDataDir();

        /** Get root data directory of user
        * \return The full path
        */
        static wxString GetUserDataDir();

        /** Get root data directory of user with the given application name
        * \return The full path
        */
        static wxString GetUserDataDir(const wxString &appName);

        /** Get documents directory of user
        * \return The full path
        */
        static wxString GetDocumentsDir();

        /** Get default user working directory (in the user data directory)
         * \return The full path
         */
        static wxString GetDefaultUserWorkingDir();

        /** Get default directory for user config (in the user data directory)
         * \return The full path
         */
        static wxString GetDefaultUserConfigDir();

    protected:

    private:

};


#endif // ASCONFIG_H
