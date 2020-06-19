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
 */

#ifndef AS_FILE_H
#define AS_FILE_H

#include "asIncludes.h"

class asFile : public wxObject {
   public:
    enum FileMode {
        ReadOnly,  // file exists, open read-only
        Write,     // file exists, open for writing
        Replace,   // create new file, even if already exists
        New,       // create new file, even if already exists
        Append     // add content to an already existing file
    };

    enum FileType { Netcdf, Grib, Text };

    explicit asFile(const wxString &fileName, const FileMode &fileMode = asFile::ReadOnly);

    ~asFile() override;

    static bool Exists(const wxString &filePath);

    bool Find();

    bool DoClose();

    virtual bool Open();

    virtual bool Close();

    bool Exists() const {
        return m_exists;
    }

   protected:
    wxFileName m_fileName;
    FileMode m_fileMode;
    bool m_exists;
    bool m_opened;

   private:
};

#endif
