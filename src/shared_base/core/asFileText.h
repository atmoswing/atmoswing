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

#ifndef AS_FILE_TEXT_H
#define AS_FILE_TEXT_H

#include <fstream>
#include <iostream>

#include "asFile.h"
#include "asIncludes.h"

/**
 * @brief Text file class.
 *
 * This class is a wrapper around the standard library.
 */
class asFileText : public asFile {
  public:
    enum FileStructType {
        ConstantWidth,
        TabsDelimited
    };

    asFileText(const wxString& fileName, const asFile::FileMode& fileMode = asFile::ReadOnly);

    ~asFileText() override = default;

    bool Open() override;

    bool Close() override;

    void AddContent(const wxString& lineContent = wxEmptyString);

    wxString GetNextLine();

    /**
     * Get the content of the file into a single string (wxString).
     */
    wxString GetContent();

    int GetInt();

    float GetFloat();

    double GetDouble();

    bool SkipLines(int linesNb);

    bool SkipElements(int elementNb);

    bool EndOfFile() const;

    static int CountLines(const wxString& filePath);

  protected:
  private:
    std::fstream m_file; /**< The file stream (not using wxTextFile because it's not optimized for files > 1Mb). */
};

#endif
