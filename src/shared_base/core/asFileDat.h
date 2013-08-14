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
 
#ifndef ASFILEDAT_H
#define ASFILEDAT_H

#include "asIncludes.h"
#include <asFileAscii.h>


class asFileDat : public asFileAscii
{
public:
/*
    //!< The file structure type
    enum ContentType
    {
        year,
        month,
        day,
        hours,
        minutes,
        data
    };
*/
    //!< Structure for pattern information
    struct Pattern
    {
        wxString Id;
        wxString Name;
        FileStructType StructType;
        int HeaderLines;
        bool ParseTime;
        int TimeYearBegin;
        int TimeYearEnd;
        int TimeMonthBegin;
        int TimeMonthEnd;
        int TimeDayBegin;
        int TimeDayEnd;
        int TimeHourBegin;
        int TimeHourEnd;
        int TimeMinuteBegin;
        int TimeMinuteEnd;
        DataParameter DataParam;
        int DataBegin;
        int DataEnd;
    };

    /** Default constructor
     * \param FileName The file path
     * \param FileMode The file access mode according to asFileAscii::FileMode
     */
    asFileDat(const wxString &FileName, const ListFileMode &FileMode);

    /** Default destructor */
    virtual ~asFileDat();

    /** Close file */
    bool Close();

    /** Load a dat file pattern defined in an xml file
     * \param FilePattern The pattern name
     * \return The file pattern
     */
    static Pattern GetPattern(const wxString &FilePatternName, const wxString &AlternatePatternDir = wxEmptyString);

    /** Get the dat file pattern max width as defined in the xml file
     * \param Pattern The pattern
     * \return The max width
     */
    static int GetPatternLineMaxCharWidth(const Pattern &Pattern);


protected:
private:

    static void InitPattern(Pattern &pattern);

    /** Convert a string to a StructType enum value
     * \param StructTypeChar The string corresponding to a StructType enum value
     * \return The StructType enum value
     */
    static FileStructType StringToStructType(const wxString &StructTypeStr);

    /** Convert a string to a ContentType enum value
     * \param ContentTypeChar The string corresponding to a ContentType enum value
     * \return The ContentType enum value
     */
//    static ContentType StringToContentType(const wxString &ContentTypeStr);

    /** Assign the pattern extracted from the file into the structure
     * \param Pattern The Pattern structure
     * \param ContentTypeStr The content type as a wxString
     * \param charstart The char number of the begining
     * \param charend The char number of the end
     */
    static bool AssignStruct(asFileDat::Pattern &Pattern, const wxString &ContentTypeStr, const int &charstart, const int &charend);
};

#endif // ASFILEDAT_H
