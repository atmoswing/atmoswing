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
 
#ifndef ASFILEASCII_H
#define ASFILEASCII_H

#include <iostream>
#include <fstream>

#include <asIncludes.h>
#include <asFile.h>

class asFileAscii : public asFile
{
    public:

    //!< The file structure type
    enum FileStructType
    {
        ConstantWidth,
        TabsDelimited
    };

    /** Default constructor
     * \param FileName The file path
     * \param fMode The file access mode according to asFileAscii::FileMode
     */
    asFileAscii(const wxString &FileName, const asFile::ListFileMode &FileMode);

    /** Default destructor */
    virtual ~asFileAscii();

    /** Open the file */
    bool Open();

    /** Closes the file */
    bool Close();

    /** Put the given content in the next line
     * \param LineContent The content to write
     */
    void AddLineContent(const wxString &LineContent = wxEmptyString);

    /** Get the next line content
     * \return The next line content
     */
    const wxString GetLineContent();

    /** Get the full file content with carriage returns
     * \return The file content
     */
    const wxString GetFullContent();

    /** Get the full file content without carriage returns
     * \return The file content on a unique line
     */
    const wxString GetFullContentWhithoutReturns();



    int GetInt();

    float GetFloat();

    double GetDouble();




    bool SkipLines(int linesNb);

    bool SkipElements(int elementNb);

    /** Check if the end of the file is reached
     * \return True if the end of the file is reached
     */
    bool EndOfFile();


protected:
private:
    std::fstream m_file;

};

#endif // ASFILEASCII_H
