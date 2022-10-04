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
 * Portions Copyright 2015 Pascal Horton, Terranum.
 */

#include <gtest/gtest.h>

#include "asFileXml.h"

TEST(FileXml, SaveAndLoadXmlFileWxStyle) {
    wxString tmpDir = asConfig::CreateTempFileName("xmlFileTest1");
    wxFileName::Mkdir(tmpDir);
    wxString filePath = tmpDir + wxFileName::GetPathSeparator() + "file.xml";

    // Write
    wxXmlDocument doc;

    wxXmlNode* nodeBase = new wxXmlNode(wxXML_ELEMENT_NODE, "base");

    wxXmlNode* nodeBuilding = new wxXmlNode(wxXML_ELEMENT_NODE, "building");
    nodeBuilding->AddAttribute("id", L"R\u00F4tillon");

    wxString stringHopital(L"h\u00F4pital", wxConvUTF8);
    ASSERT_TRUE(stringHopital.size() > 0);
    wxXmlNode* nodeType = new wxXmlNode(wxXML_ELEMENT_NODE, "building_type");
    wxXmlNode* nodeTypeValue = new wxXmlNode(wxXML_TEXT_NODE, "building_type", stringHopital);
    nodeType->AddChild(nodeTypeValue);

    wxXmlNode* nodeLocation = new wxXmlNode(wxXML_ELEMENT_NODE, "building_location");
    wxXmlNode* nodeLocationValue = new wxXmlNode(wxXML_TEXT_NODE, "building_location", L"Z\u00FCrich");
    nodeLocation->AddChild(nodeLocationValue);

    wxXmlNode* nodeHeight = new wxXmlNode(wxXML_ELEMENT_NODE, "building_height");
    wxXmlNode* nodeHeightValue = new wxXmlNode(wxXML_TEXT_NODE, "building_height", "40");
    nodeHeight->AddChild(nodeHeightValue);

    nodeBuilding->AddChild(nodeType);
    nodeBuilding->AddChild(nodeLocation);
    nodeBuilding->AddChild(nodeHeight);

    nodeBase->AddChild(nodeBuilding);

    doc.SetRoot(nodeBase);

    doc.Save(filePath);

    // Read
    wxXmlDocument doc2;

    ASSERT_TRUE(doc2.Load(filePath));

    EXPECT_EQ("base", doc2.GetRoot()->GetName());

    wxXmlNode* childBuilding = doc2.GetRoot()->GetChildren();
    EXPECT_EQ(L"R\u00F4tillon", childBuilding->GetAttribute("id"));
    EXPECT_TRUE(childBuilding->GetAttribute("id").size() > 0);

    wxXmlNode* childBuildingType = childBuilding->GetChildren();
    EXPECT_EQ("building_type", childBuildingType->GetName());
    EXPECT_EQ(L"h\u00F4pital", childBuildingType->GetNodeContent());
    EXPECT_TRUE(childBuildingType->GetNodeContent().size() > 0);

    wxXmlNode* childBuildingLocation = childBuildingType->GetNext();
    EXPECT_EQ("building_location", childBuildingLocation->GetName());
    EXPECT_EQ(L"Z\u00FCrich", childBuildingLocation->GetNodeContent());

    wxXmlNode* childBuildingHeight = childBuildingLocation->GetNext();
    EXPECT_EQ("building_height", childBuildingHeight->GetName());
    EXPECT_EQ("40", childBuildingHeight->GetNodeContent());

    asRemoveDir(tmpDir);
}

TEST(FileXml, SaveAndLoadXmlFileAtmoSwingStyle) {
    wxString tmpDir = asConfig::CreateTempFileName("xmlFileTest2");
    wxFileName::Mkdir(tmpDir);
    wxString filePath = tmpDir + wxFileName::GetPathSeparator() + "file2.xml";

    wxString stringHopital(L"h\u00F4pital", wxConvUTF8);
    ASSERT_TRUE(stringHopital.size() > 0);

    // Write
    asFileXml fileXml(filePath, asFile::Replace);
    ASSERT_TRUE(fileXml.Open());

    wxXmlNode* nodeBuilding = new wxXmlNode(wxXML_ELEMENT_NODE, "building");
    nodeBuilding->AddAttribute("id", wxString(L"R\u00F4tillon", wxConvUTF8));

    nodeBuilding->AddChild(fileXml.CreateNodeWithValue("building_type", stringHopital));
    nodeBuilding->AddChild(fileXml.CreateNodeWithValue("building_location", wxString(L"Z\u00FCrich", wxConvUTF8)));
    nodeBuilding->AddChild(fileXml.CreateNodeWithValue("building_height", 40));

    fileXml.AddChild(nodeBuilding);

    fileXml.Save();

    // Read
    asFileXml fileXml2(filePath, asFile::ReadOnly);
    ASSERT_TRUE(fileXml2.Open());

    EXPECT_EQ("atmoswing", fileXml2.GetRoot()->GetName());

    wxXmlNode* childBuilding = fileXml2.GetRoot()->GetChildren();
    EXPECT_EQ(L"R\u00F4tillon", childBuilding->GetAttribute("id"));
    EXPECT_TRUE(childBuilding->GetAttribute("id").size() > 0);

    wxXmlNode* childBuildingType = childBuilding->GetChildren();
    EXPECT_EQ("building_type", childBuildingType->GetName());
    EXPECT_EQ(stringHopital, childBuildingType->GetNodeContent());
    EXPECT_TRUE(childBuildingType->GetNodeContent().size() > 0);

    wxXmlNode* childBuildingLocation = childBuildingType->GetNext();
    EXPECT_EQ("building_location", childBuildingLocation->GetName());
    EXPECT_EQ(L"Z\u00FCrich", childBuildingLocation->GetNodeContent());

    wxXmlNode* childBuildingHeight = childBuildingLocation->GetNext();
    EXPECT_EQ("building_height", childBuildingHeight->GetName());
    EXPECT_EQ("40", childBuildingHeight->GetNodeContent());

    asRemoveDir(tmpDir);
}

TEST(FileXml, LoadSimpleXmlFile) {
    wxString filePath = wxFileName::GetCwd();
    filePath.Append("/files/file_xml.xml");
    asFileXml fileXml(filePath, asFile::ReadOnly);

    EXPECT_TRUE(fileXml.Open());
}

TEST(FileXml, LoadSimpleXmlFileWithErrors) {
    wxLogNull logNo;

    wxString filePath = wxFileName::GetCwd();
    filePath.Append("/files/file_xml_error.xml");
    asFileXml fileXml(filePath, asFile::ReadOnly);

    EXPECT_FALSE(fileXml.Open());
}
