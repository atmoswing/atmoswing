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
 
#include "asFileDat.h"

#include <asCatalog.h>
#include <asFileXml.h>


asFileDat::asFileDat(const wxString &FileName, const ListFileMode &FileMode)
:
asFileAscii(FileName, FileMode)
{

}

asFileDat::~asFileDat()
{
    //dtor
}

bool asFileDat::Close()
{
    wxASSERT(m_Opened);

    return true;
}

void asFileDat::InitPattern(asFileDat::Pattern &pattern)
{
    pattern.HeaderLines = 0;
    pattern.ParseTime = false;
    pattern.TimeYearBegin = 0;
    pattern.TimeYearEnd = 0;
    pattern.TimeMonthBegin = 0;
    pattern.TimeMonthEnd = 0;
    pattern.TimeDayBegin = 0;
    pattern.TimeDayEnd = 0;
    pattern.TimeHourBegin = 0;
    pattern.TimeHourEnd = 0;
    pattern.TimeMinuteBegin = 0;
    pattern.TimeMinuteEnd = 0;
    pattern.DataBegin = 0;
    pattern.DataEnd = 0;
}

asFileDat::Pattern asFileDat::GetPattern(const wxString &FilePatternName, const wxString &AlternatePatternDir)
{
    asFileDat::Pattern pattern;

    InitPattern(pattern);

    // Load xml file
    wxString FileName;
    if (!AlternatePatternDir.IsEmpty())
    {
        FileName = AlternatePatternDir + DS + FilePatternName + ".xml";
    }
    else
    {
        ThreadsManager().CritSectionConfig().Enter();
        wxString PatternsDir = wxFileConfig::Get()->Read("/PredictandDBToolbox/PatternsDir", wxEmptyString);
        ThreadsManager().CritSectionConfig().Leave();
        FileName = PatternsDir + DS + FilePatternName + ".xml";
    }
    asFileXml xmlFile( FileName, asFile::ReadOnly );
    if(!xmlFile.Open()) return pattern;

    // XML struct for the dataset information
    wxString PatternAccess = "pattern";
    wxString PatternTimeBlockAccess = "time.block";
    wxString PatternDataBlockAccess = "data.block";

    // Set the base node
    if(!xmlFile.GoToFirstNodeWithPath(PatternAccess)) return pattern;

    // Get the pattern informations
    pattern.Id = xmlFile.GetFirstElementAttributeValueText(wxEmptyString, "id", wxEmptyString);
    pattern.Name = xmlFile.GetFirstElementAttributeValueText(wxEmptyString, "name", wxEmptyString);

    // Get file structure
    wxString tmpStructType = xmlFile.GetFirstElementValueText("structtype", wxEmptyString);
    pattern.StructType = StringToStructType(tmpStructType);
    pattern.HeaderLines = xmlFile.GetFirstElementValueInt("headerlines", 0);
    pattern.ParseTime = xmlFile.GetFirstElementValueBool("parsetime", true);

    // Different tags if it's a constant width or tab delimited file
    wxString tagstart, tagend;
    switch (pattern.StructType)
    {
        case (asFileDat::ConstantWidth):
            tagstart = "charstart";
            tagend = "charend";
            break;
        case (asFileDat::TabsDelimited):
            tagstart = "column";
            tagend = "column";
            break;
        default:
            asThrowException(_("The file structure type in unknown"));
    }

    // Get first time block element
    wxString tmpTimeBlockContent = xmlFile.GetFirstElementValueText(PatternTimeBlockAccess, wxEmptyString);

    int charstart = xmlFile.GetThisElementAttributeValueInt(tagstart, 0);
    int charend = xmlFile.GetThisElementAttributeValueInt(tagend, 0);

    //AssignStruct(Pattern, tmpTimeBlockContent, xmlFile.GetFirstElementAttributeValueInt(PatternTimeBlockAccess, tagstart, 0), xmlFile.GetFirstElementAttributeValueInt(PatternTimeBlockAccess, tagend, 0))
    asFileDat::AssignStruct(pattern, tmpTimeBlockContent, charstart, charend);

    // Get other time block elements
    while(xmlFile.GetNextElement(PatternTimeBlockAccess))
    {
        tmpTimeBlockContent = xmlFile.GetThisElementValueText(wxEmptyString);
        int charstart = xmlFile.GetThisElementAttributeValueInt(tagstart, 0);
        int charend = xmlFile.GetThisElementAttributeValueInt(tagend, 0);
        AssignStruct(pattern, tmpTimeBlockContent, charstart, charend);
    }

    // Get first data block element
    wxString tmpDataBlockContent = xmlFile.GetFirstElementValueText(PatternDataBlockAccess, wxEmptyString);
    pattern.DataParam = asGlobEnums::StringToDataParameterEnum(tmpDataBlockContent);
    pattern.DataBegin = xmlFile.GetFirstElementAttributeValueInt(PatternDataBlockAccess, tagstart, 0);
    pattern.DataEnd = xmlFile.GetFirstElementAttributeValueInt(PatternDataBlockAccess, tagend, 0);

    // Reset the base path
    xmlFile.ClearCurrenNodePath();

    return pattern;
}

asFileDat::FileStructType asFileDat::StringToStructType(const wxString &StructTypeStr)
{
    if (StructTypeStr.CmpNoCase("tabsdelimited")==0) {return asFileDat::TabsDelimited;}
    else if (StructTypeStr.CmpNoCase("constantwidth")==0) {return asFileDat::ConstantWidth;}
    else {asThrowException(_("The file structure type in unknown"));}
}

bool asFileDat::AssignStruct(asFileDat::Pattern &Pattern, const wxString &ContentTypeStr, const int &charstart, const int &charend)
{
    if ((ContentTypeStr.CmpNoCase("year")==0) | (ContentTypeStr.CmpNoCase("years")==0))
    {
        Pattern.TimeYearBegin = charstart;
        Pattern.TimeYearEnd = charend;
        return true;
    }
    else if ((ContentTypeStr.CmpNoCase("month")==0) | (ContentTypeStr.CmpNoCase("months")==0))
    {
        Pattern.TimeMonthBegin = charstart;
        Pattern.TimeMonthEnd = charend;
        return true;
    }
    else if ((ContentTypeStr.CmpNoCase("day")==0) | (ContentTypeStr.CmpNoCase("days")==0))
    {
        Pattern.TimeDayBegin = charstart;
        Pattern.TimeDayEnd = charend;
        return true;
    }
    else if ((ContentTypeStr.CmpNoCase("hour")==0) | (ContentTypeStr.CmpNoCase("hours")==0))
    {
        Pattern.TimeHourBegin = charstart;
        Pattern.TimeHourEnd = charend;
        return true;
    }
    else if ((ContentTypeStr.CmpNoCase("minute")==0) | (ContentTypeStr.CmpNoCase("minutes")==0))
    {
        Pattern.TimeMinuteBegin = charstart;
        Pattern.TimeMinuteEnd = charend;
        return true;
    }
    else
    {
        asThrowException(_("The content type in unknown"));
    }
}

int asFileDat::GetPatternLineMaxCharWidth(const asFileDat::Pattern &Pattern)
{
    int maxwidth = 0;

    maxwidth = wxMax(maxwidth, Pattern.TimeYearEnd);
    maxwidth = wxMax(maxwidth, Pattern.TimeMonthEnd);
    maxwidth = wxMax(maxwidth, Pattern.TimeDayEnd);
    maxwidth = wxMax(maxwidth, Pattern.TimeHourEnd);
    maxwidth = wxMax(maxwidth, Pattern.TimeMinuteEnd);
    maxwidth = wxMax(maxwidth, Pattern.DataEnd);

    return maxwidth;
}
