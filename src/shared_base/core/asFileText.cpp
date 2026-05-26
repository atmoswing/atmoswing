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

#include "asFileText.h"
#include "asIncludes.h"

asFileText::asFileText(const wxString& fileName, const FileMode& fileMode)
    : asFile(fileName, fileMode) {}

bool asFileText::Open() {
    if (!Find()) return false;

    switch (_fileMode) {
        case (ReadOnly):
            _file.open(_fileName.GetFullPath().mb_str(), std::fstream::in);
            break;

        case (Write):
            _file.open(_fileName.GetFullPath().mb_str(), std::fstream::out);
            break;

        case (Replace):
            _file.open(_fileName.GetFullPath().mb_str(), std::fstream::trunc | std::fstream::out);
            break;

        case (New):
            _file.open(_fileName.GetFullPath().mb_str(), std::fstream::out);
            break;

        case (Append):
            _file.open(_fileName.GetFullPath().mb_str(), std::fstream::app | std::fstream::out);
            break;
    }

    if (!_file.is_open()) return false;

    _opened = true;

    return true;
}

bool asFileText::Close() {
    wxASSERT(_opened);

    _file.close();
    return true;
}

void asFileText::AddContent(const wxString& lineContent) {
    wxASSERT(_opened);

    _file << lineContent.mb_str();

    // Check the state flags
    if (_file.fail())
        throw std::runtime_error(asStrF(_("An error occured while trying to write in file %s"), _fileName.GetFullPath()));
}

wxString asFileText::GetNextLine() {
    wxASSERT(_opened);

    std::string tmpLineContent;

    if (!_file.eof()) {
        getline(_file, tmpLineContent);

        // Check the state flags
        if ((!_file.eof()) && (_file.fail()))
            throw std::runtime_error(
                asStrF(_("An error occured while trying to write in file %s"), _fileName.GetFullPath()));
    } else {
        throw std::runtime_error(
            asStrF(_("You are trying to read a line after the end of the file %s"), _fileName.GetFullPath()));
    }

    wxString lineContent = wxString(tmpLineContent.c_str(), wxConvUTF8);

    return lineContent;
}

wxString asFileText::GetContent() {
    wxString content;

    while (!EndOfFile()) {
        content.Append(GetNextLine() + "\n");
    }

    return content;
}

int asFileText::GetInt() {
    wxASSERT(_opened);

    int tmp;
    _file >> tmp;
    return tmp;
}

float asFileText::GetFloat() {
    wxASSERT(_opened);

    float tmp;
    _file >> tmp;
    return tmp;
}

double asFileText::GetDouble() {
    wxASSERT(_opened);

    double tmp;
    _file >> tmp;
    return tmp;
}

bool asFileText::SkipLines(int linesNb) {
    wxASSERT(_opened);

    for (int iLine = 0; iLine < linesNb; iLine++) {
        if (!_file.eof()) {
            const wxString skippedLine = GetNextLine();
            (void)skippedLine;
        } else {
            wxLogError(_("Reached the end of the file while skipping lines."));
            return false;
        }
    }

    return true;
}

bool asFileText::SkipElements(int elementNb) {
    wxASSERT(_opened);

    float tmp;

    for (int iEl = 0; iEl < elementNb; iEl++) {
        if (!_file.eof()) {
            _file >> tmp;
        } else {
            wxLogError(_("Reached the end of the file while skipping lines."));
            return false;
        }
    }

    return true;
}

bool asFileText::EndOfFile() const {
    wxASSERT(_opened);

    return _file.eof();
}

int asFileText::CountLines(const wxString& filePath) {
    asFileText file(filePath, asFile::ReadOnly);
    if (!file.Open()) {
        wxLogError(_("Couldn't open the file %s."), filePath);
        return 0;
    }

    int lines = 0;
    wxString content;
    do {
        content = file.GetNextLine();
        if (content.Length() > 0) {
            lines++;
        }
    } while (!file.EndOfFile());
    if (!file.Close()) {
        wxLogVerbose(_("Couldn't properly close the file %s after counting lines."), filePath);
    }

    return lines;
}
