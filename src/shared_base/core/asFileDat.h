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
 
#ifndef ASFILEDAT_H
#define ASFILEDAT_H

#include "asIncludes.h"
#include <asFileAscii.h>


class asFileDat : public asFileAscii
{
public:

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

};

#endif // ASFILEDAT_H
