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
        ReadOnly,    // file exists, open read-only
        Write,        // file exists, open for writing
        Replace,    // create new file, even if already exists
        New,        // create new file, even if already exists
        Append        // add content to an already existing file
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
