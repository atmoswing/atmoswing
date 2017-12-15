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

#include "asFileDat.h"

#include <asFileXml.h>


asFileDat::asFileDat(const wxString &FileName, const ListFileMode &FileMode)
        : asFileAscii(FileName, FileMode)
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
    pattern.headerLines = 0;
    pattern.parseTime = false;
    pattern.timeYearBegin = 0;
    pattern.timeYearEnd = 0;
    pattern.timeMonthBegin = 0;
    pattern.timeMonthEnd = 0;
    pattern.timeDayBegin = 0;
    pattern.timeDayEnd = 0;
    pattern.timeHourBegin = 0;
    pattern.timeHourEnd = 0;
    pattern.timeMinuteBegin = 0;
    pattern.timeMinuteEnd = 0;
    pattern.dataBegin = 0;
    pattern.dataEnd = 0;
}

asFileDat::Pattern asFileDat::GetPattern(const wxString &fileName, const wxString &directory)
{
    asFileDat::Pattern pattern;

    InitPattern(pattern);

    // Load xml file
    wxString FileName;
    if (!directory.IsEmpty()) {
        FileName = directory + DS + fileName + ".xml";
    } else {
        ThreadsManager().CritSectionConfig().Enter();
        wxString PatternsDir = wxFileConfig::Get()->Read("/PredictandDBToolbox/PatternsDir", wxEmptyString);
        ThreadsManager().CritSectionConfig().Leave();
        FileName = PatternsDir + DS + fileName + ".xml";
    }

    asFileXml xmlFile(FileName, asFile::ReadOnly);
    if (!xmlFile.Open()) {
        asThrowException(_("Cannot open the pattern file."));
    }
    if (!xmlFile.CheckRootElement()) {
        asThrowException(_("Errors were found in the pattern file."));
    }

    // Get data
    long charStart, charEnd;
    wxString charStartStr, charEndStr, attributeStart, attributeEnd;
    wxXmlNode *node = xmlFile.GetRoot()->GetChildren();
    if (node->GetName() == "pattern") {
        pattern.id = node->GetAttribute("id");
        pattern.name = node->GetAttribute("name");

        wxXmlNode *nodeParam = node->GetChildren();
        while (nodeParam) {
            if (nodeParam->GetName() == "structure_type") {
                pattern.structType = StringToStructType(xmlFile.GetString(nodeParam));
                switch (pattern.structType) {
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
                pattern.headerLines = xmlFile.GetInt(nodeParam);
            } else if (nodeParam->GetName() == "parse_time") {
                pattern.parseTime = xmlFile.GetBool(nodeParam);
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
                        pattern.timeYearBegin = charStart;
                        pattern.timeYearEnd = charEnd;
                    } else if (nodeTime->GetName() == "month") {
                        pattern.timeMonthBegin = charStart;
                        pattern.timeMonthEnd = charEnd;
                    } else if (nodeTime->GetName() == "day") {
                        pattern.timeDayBegin = charStart;
                        pattern.timeDayEnd = charEnd;
                    } else if (nodeTime->GetName() == "hour") {
                        pattern.timeHourBegin = charStart;
                        pattern.timeHourEnd = charEnd;
                    } else if (nodeTime->GetName() == "minute") {
                        pattern.timeMinuteBegin = charStart;
                        pattern.timeMinuteEnd = charEnd;
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
                        pattern.dataBegin = charStart;
                        pattern.dataEnd = charEnd;
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
    if (StructTypeStr.CmpNoCase("tabs_delimited") == 0) {
        return asFileDat::TabsDelimited;
    } else if (StructTypeStr.CmpNoCase("constant_width") == 0) {
        return asFileDat::ConstantWidth;
    } else {
        asThrowException(_("The file structure type in unknown"));
    }
}

int asFileDat::GetPatternLineMaxCharWidth(const asFileDat::Pattern &Pattern)
{
    int maxwidth = 0;

    maxwidth = wxMax(maxwidth, Pattern.timeYearEnd);
    maxwidth = wxMax(maxwidth, Pattern.timeMonthEnd);
    maxwidth = wxMax(maxwidth, Pattern.timeDayEnd);
    maxwidth = wxMax(maxwidth, Pattern.timeHourEnd);
    maxwidth = wxMax(maxwidth, Pattern.timeMinuteEnd);
    maxwidth = wxMax(maxwidth, Pattern.dataEnd);

    return maxwidth;
}
