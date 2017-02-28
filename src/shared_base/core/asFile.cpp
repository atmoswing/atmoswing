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

#include "asFile.h"

bool asRemoveDir(const wxString &Path)
{
    wxString f = wxFindFirstFile(Path + "/*.*");
    while (!f.empty()) {
        if (!wxRemoveFile(f))
            return false;
        f = wxFindNextFile();
    }

    return wxRmdir(Path);

}

asFile::asFile(const wxString &FileName, const ListFileMode &FileMode)
{
    m_fileName = wxFileName(FileName);
    m_fileMode = FileMode;
    m_exists = false;
    m_opened = false;
}

asFile::~asFile()
{
    DoClose();
}

bool asFile::Exists(const wxString &FilePath)
{
    return wxFileName::FileExists(FilePath);
}

bool asFile::Find()
{
    bool missingFile = false, missingDir = false, mkDir = false, errorRights = false, errorOverwrite = false;

    if (!m_fileName.IsOk()) {
        wxLogError(_("The file path is not OK %s"), m_fileName.GetFullPath());
        return false;
    }

    switch (m_fileMode) {
        case (ReadOnly):
            if (!wxFileName::FileExists(m_fileName.GetFullPath())) {
                missingFile = true;
            } else {
                m_exists = true;
                if (!wxFileName::IsFileReadable(m_fileName.GetFullPath())) {
                    errorRights = true;
                }
            }
            break;

        case (Write):
            if (!wxFileName::FileExists(m_fileName.GetFullPath())) {
                if (!wxFileName::DirExists(m_fileName.GetPath())) {
                    mkDir = true;
                } else {
                    if (!wxFileName::IsDirWritable(m_fileName.GetPath())) {
                        missingDir = true;
                    }
                }
            } else {
                m_exists = true;
                if (!wxFileName::IsFileWritable(m_fileName.GetFullPath())) {
                    errorRights = true;
                }
            }
            break;

        case (Replace):
            if (!wxFileName::FileExists(m_fileName.GetFullPath())) {
                if (!wxFileName::DirExists(m_fileName.GetPath())) {
                    mkDir = true;
                } else {
                    if (!wxFileName::IsDirWritable(m_fileName.GetPath())) {
                        missingDir = true;
                    }
                }
            } else {
                m_exists = true;
                if (!wxFileName::IsFileWritable(m_fileName.GetFullPath())) {
                    errorRights = true;
                }
            }
            break;

        case (New):
            if (!wxFileName::FileExists(m_fileName.GetFullPath())) {
                if (!wxFileName::DirExists(m_fileName.GetPath())) {
                    mkDir = true;
                } else {
                    if (!wxFileName::IsDirWritable(m_fileName.GetPath())) {
                        missingDir = true;
                    }
                }
            } else {
                m_exists = true;
                errorOverwrite = true;
            }
            break;

        case (Append):
            if (!wxFileName::FileExists(m_fileName.GetFullPath())) {
                missingFile = true;
            } else {
                m_exists = true;
                if (!wxFileName::IsFileWritable(m_fileName.GetFullPath())) {
                    errorRights = true;
                }
            }
            break;

        default :
            asThrowException(_("The file access is not correctly set."));
    }

    if (mkDir) {
        if (!wxFileName::Mkdir(m_fileName.GetPath(), wxS_DIR_DEFAULT, wxPATH_MKDIR_FULL)) {
            wxLogError(_("The directory %s could not be created."), m_fileName.GetPath());
            return false;
        }
        m_exists = true;
    }

    if (missingFile) {
        wxLogError(_("Cannot find the file %s"), m_fileName.GetFullPath());
        return false;
    }

    if (missingDir) {
        wxLogError(_("Cannot find the directory %s"), m_fileName.GetFullPath());
        return false;
    }

    if (errorRights) {
        wxLogError(_("The file could not be accessed in the desired mode."));
        return false;
    }

    if (errorOverwrite) {
        wxLogError(_("The file should be overwritten, which is not allowed in the New mode."));
        return false;
    }

    return true;
}

bool asFile::DoClose()
{
    Close();
    return true;
}

bool asFile::Open()
{
    return false;
}

bool asFile::Close()
{
    return false;
}

