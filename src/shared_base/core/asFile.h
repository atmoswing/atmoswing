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
 
#ifndef ASFILE_H
#define ASFILE_H

#include "asIncludes.h"

bool asRemoveDir(const wxString &Path);

class asFile: public wxObject
{
public:

    //!< The file access mode
    enum ListFileMode
    {
        ReadOnly,	// file exists, open read-only
        Write,		// file exists, open for writing
        Replace,	// create new file, even if already exists
        New,	    // create new file, even if already exists
        Append	    // add content to an already existing file
    };

    /** Default constructor */
    asFile(const wxString &FileName, const ListFileMode &FileMode = asFile::ReadOnly);

    /** Default destructor */
    virtual ~asFile();

    /** Check if the file exists */
    static bool Exists(const wxString &FilePath);

    /** Check for the file existance */
    bool Find();

    /** Trigger the close file */
    bool DoClose();

    /** Open file */
    virtual bool Open();

    /** Close file */
    virtual bool Close();

    /** Close file */
    bool Exists()
    {
        return m_Exists;
    }


protected:
    ListFileMode m_FileMode;
    wxFileName m_FileName;
    bool m_Exists;
    bool m_Opened;

private:
};

#endif // ASFILE_H
