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

#include "asFileXml.h"

asFileDat::asFileDat(const wxString& fileName, const FileMode& fileMode)
    : asFileText(fileName, fileMode) {}

bool asFileDat::Close() {
    wxASSERT(m_opened);

    return true;
}

void asFileDat::InitPattern(asFileDat::Pattern& pattern) {
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

asFileDat::Pattern asFileDat::GetPattern(const wxString& fileName, const wxString& directory) {
    asFileDat::Pattern pattern;

    InitPattern(pattern);

    // Load xml file
    wxString filePath;
    if (!directory.IsEmpty()) {
        filePath = directory + DS + fileName + ".xml";
    } else {
        ThreadsManager().CritSectionConfig().Enter();
        wxString patternsDir = wxFileConfig::Get()->Read("/PredictandDBToolbox/PatternsDir", wxEmptyString);
        ThreadsManager().CritSectionConfig().Leave();
        filePath = patternsDir + DS + fileName + ".xml";
    }

    asFileXml xmlFile(filePath, asFile::ReadOnly);
    if (!xmlFile.Open()) {
        throw runtime_error(_("Cannot open the pattern file."));
    }
    if (!xmlFile.CheckRootElement()) {
        throw runtime_error(_("Errors were found in the pattern file."));
    }

    // Get data
    long charStart, charEnd;
    wxString charStartStr, charEndStr, attributeStart, attributeEnd;
    wxXmlNode* node = xmlFile.GetRoot()->GetChildren();
    if (node->GetName() == "pattern") {
        pattern.id = node->GetAttribute("id");
        pattern.name = node->GetAttribute("name");

        wxXmlNode* nodeParam = node->GetChildren();
        while (nodeParam) {
            if (nodeParam->GetName() == "structure_type") {
                pattern.structType = StringToStructType(asFileXml::GetString(nodeParam));
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
                        throw runtime_error(_("The file structure type in unknown"));
                }
            } else if (nodeParam->GetName() == "header_lines") {
                pattern.headerLines = asFileXml::GetInt(nodeParam);
            } else if (nodeParam->GetName() == "parse_time") {
                pattern.parseTime = asFileXml::GetBool(nodeParam);
            } else if (nodeParam->GetName() == "time") {
                if (attributeStart.IsEmpty() || attributeEnd.IsEmpty()) {
                    throw runtime_error(_("The file structure type in undefined"));
                }

                wxXmlNode* nodeTime = nodeParam->GetChildren();
                while (nodeTime) {
                    charStartStr = nodeTime->GetAttribute(attributeStart);
                    charStartStr.ToLong(&charStart);
                    charEndStr = nodeTime->GetAttribute(attributeEnd);
                    charEndStr.ToLong(&charEnd);

                    if (nodeTime->GetName() == "year") {
                        pattern.timeYearBegin = (int)charStart;
                        pattern.timeYearEnd = (int)charEnd;
                    } else if (nodeTime->GetName() == "month") {
                        pattern.timeMonthBegin = (int)charStart;
                        pattern.timeMonthEnd = (int)charEnd;
                    } else if (nodeTime->GetName() == "day") {
                        pattern.timeDayBegin = (int)charStart;
                        pattern.timeDayEnd = (int)charEnd;
                    } else if (nodeTime->GetName() == "hour") {
                        pattern.timeHourBegin = (int)charStart;
                        pattern.timeHourEnd = (int)charEnd;
                    } else if (nodeTime->GetName() == "minute") {
                        pattern.timeMinuteBegin = (int)charStart;
                        pattern.timeMinuteEnd = (int)charEnd;
                    } else {
                        xmlFile.UnknownNode(nodeTime);
                    }

                    nodeTime = nodeTime->GetNext();
                }
            } else if (nodeParam->GetName() == "data") {
                if (attributeStart.IsEmpty() || attributeEnd.IsEmpty()) {
                    throw runtime_error(_("The file structure type in undefined"));
                }

                wxXmlNode* nodeData = nodeParam->GetChildren();
                while (nodeData) {
                    charStartStr = nodeData->GetAttribute(attributeStart);
                    charStartStr.ToLong(&charStart);
                    charEndStr = nodeData->GetAttribute(attributeEnd);
                    charEndStr.ToLong(&charEnd);

                    if (nodeData->GetName() == "value") {
                        pattern.dataBegin = (int)charStart;
                        pattern.dataEnd = (int)charEnd;
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
        throw runtime_error(_("Expecting the tag pattern in the pattern file..."));
    }

    return pattern;
}

asFileDat::FileStructType asFileDat::StringToStructType(const wxString& structTypeStr) {
    if (structTypeStr.CmpNoCase("tabs_delimited") == 0) {
        return asFileDat::TabsDelimited;
    } else if (structTypeStr.CmpNoCase("constant_width") == 0) {
        return asFileDat::ConstantWidth;
    } else {
        throw runtime_error(_("The file structure type in unknown"));
    }
}

int asFileDat::GetPatternLineMaxCharWidth(const asFileDat::Pattern& pattern) {
    int maxwidth = 0;

    maxwidth = wxMax(maxwidth, pattern.timeYearEnd);
    maxwidth = wxMax(maxwidth, pattern.timeMonthEnd);
    maxwidth = wxMax(maxwidth, pattern.timeDayEnd);
    maxwidth = wxMax(maxwidth, pattern.timeHourEnd);
    maxwidth = wxMax(maxwidth, pattern.timeMinuteEnd);
    maxwidth = wxMax(maxwidth, pattern.dataEnd);

    return maxwidth;
}
