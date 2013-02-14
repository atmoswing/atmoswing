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
    std::fstream m_File;

};

#endif // ASFILEASCII_H
