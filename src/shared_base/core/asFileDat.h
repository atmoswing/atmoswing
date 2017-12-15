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
 * Portions Copyright 2013-2015 Pascal Horton, Terranum.
 */

#ifndef ASFILEDAT_H
#define ASFILEDAT_H

#include "asIncludes.h"
#include <asFileAscii.h>


class asFileDat
        : public asFileAscii
{
public:
    struct Pattern
    {
        wxString id;
        wxString name;
        FileStructType structType;
        int headerLines;
        bool parseTime;
        int timeYearBegin;
        int timeYearEnd;
        int timeMonthBegin;
        int timeMonthEnd;
        int timeDayBegin;
        int timeDayEnd;
        int timeHourBegin;
        int timeHourEnd;
        int timeMinuteBegin;
        int timeMinuteEnd;
        int dataBegin;
        int dataEnd;
    };

    asFileDat(const wxString &FileName, const ListFileMode &FileMode);

    virtual ~asFileDat();

    bool Close();

    static Pattern GetPattern(const wxString &fileName, const wxString &directory = wxEmptyString);

    static int GetPatternLineMaxCharWidth(const Pattern &Pattern);

protected:

private:
    static void InitPattern(Pattern &pattern);

    static FileStructType StringToStructType(const wxString &StructTypeStr);
};

#endif // ASFILEDAT_H
