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
    wxASSERT(m_opened);

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
    if(!xmlFile.Open())
    {
        asThrowException(_("Cannot open the pattern file."));
    }
    if(!xmlFile.CheckRootElement())
    {
        asThrowException(_("Errors were found in the pattern file."));
    }
    
    // Get data
    long charStart, charEnd;
    wxString charStartStr, charEndStr, attributeStart, attributeEnd;
    wxXmlNode *node = xmlFile.GetRoot()->GetChildren();
    if (node->GetName() == "pattern") {
        pattern.Id = node->GetAttribute("id");
        pattern.Name = node->GetAttribute("name");

        wxXmlNode *nodeParam = node->GetChildren();
        while (nodeParam) {
            if (nodeParam->GetName() == "structure_type") {
                pattern.StructType = StringToStructType(xmlFile.GetString(nodeParam));
                switch (pattern.StructType)
                {
                    case (asFileDat::ConstantWidth):
                        attributeStart = "char_start";
                        attributeEnd = "char_end";
                        break;
                    case (asFileDat::TabsDelimited):
                        attributeStart = "column";
                        attributeEnd = "column";
                        break;
                    default:
                        asThrowException(_("The file structure type in unknown"));
                }
            } else if (nodeParam->GetName() == "header_lines") {
                pattern.HeaderLines = xmlFile.GetInt(nodeParam);
            } else if (nodeParam->GetName() == "parse_time") {
                pattern.ParseTime = xmlFile.GetBool(nodeParam);
            } else if (nodeParam->GetName() == "time") {
                if (attributeStart.IsEmpty() || attributeEnd.IsEmpty()) {
                    asThrowException(_("The file structure type in undefined"));
                }

                wxXmlNode *nodeTime = nodeParam->GetChildren();
                while (nodeTime) {
                    
                    charStartStr = nodeTime->GetAttribute(attributeStart);
                    charStartStr.ToLong(&charStart);
                    charEndStr = nodeTime->GetAttribute(attributeEnd);
                    charEndStr.ToLong(&charEnd);

                    if (nodeTime->GetName() == "year") {
                        pattern.TimeYearBegin = charStart;
                        pattern.TimeYearEnd = charEnd;
                    } else if (nodeTime->GetName() == "month") {
                        pattern.TimeMonthBegin = charStart;
                        pattern.TimeMonthEnd = charEnd;
                    } else if (nodeTime->GetName() == "day") {
                        pattern.TimeDayBegin = charStart;
                        pattern.TimeDayEnd = charEnd;
                    } else if (nodeTime->GetName() == "hour") {
                        pattern.TimeHourBegin = charStart;
                        pattern.TimeHourEnd = charEnd;
                    } else if (nodeTime->GetName() == "minute") {
                        pattern.TimeMinuteBegin = charStart;
                        pattern.TimeMinuteEnd = charEnd;
                    } else {
                        xmlFile.UnknownNode(nodeTime);
                    }
                    
                    nodeTime = nodeTime->GetNext();
                }
            } else if (nodeParam->GetName() == "data") {
                if (attributeStart.IsEmpty() || attributeEnd.IsEmpty()) {
                    asThrowException(_("The file structure type in undefined"));
                }

                wxXmlNode *nodeData = nodeParam->GetChildren();
                while (nodeData) {

                    charStartStr = nodeData->GetAttribute(attributeStart);
                    charStartStr.ToLong(&charStart);
                    charEndStr = nodeData->GetAttribute(attributeEnd);
                    charEndStr.ToLong(&charEnd);

                    if (nodeData->GetName() == "value") {
                        pattern.DataBegin = charStart;
                        pattern.DataEnd = charEnd;
                    } else {
                        xmlFile.UnknownNode(nodeData);
                    }

                    nodeData = nodeData->GetNext();
                }
            } else {
                xmlFile.UnknownNode(nodeParam);
            }

            nodeParam = nodeParam->GetNext();
        }

    } else {
        asThrowException(_("Expecting the tag pattern in the pattern file..."));
    }

    return pattern;
}

asFileDat::FileStructType asFileDat::StringToStructType(const wxString &StructTypeStr)
{
    if (StructTypeStr.CmpNoCase("tabs_delimited")==0) {
        return asFileDat::TabsDelimited;
    }
    else if (StructTypeStr.CmpNoCase("constant_width")==0) {
        return asFileDat::ConstantWidth;
    }
    else {
        asThrowException(_("The file structure type in unknown"));
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
