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
 * Portions Copyright 2015 Pascal Horton, Terr@num.
 */

#include "include_tests.h"
#include "asFileXml.h"

#include "UnitTest++.h"

namespace
{
    
TEST(SaveAndLoadXmlFileWxStyle)
{
    wxString tmpDir = asConfig::CreateTempFileName("xmlFileTest");
    wxFileName::Mkdir(tmpDir);
    wxString filePath = tmpDir + wxFileName::GetPathSeparator() + "file.xml";

    // Write
    wxXmlDocument doc;

    wxXmlNode * nodeBase = new wxXmlNode(wxXML_ELEMENT_NODE ,"base");

    wxXmlNode * nodeBuilding = new wxXmlNode(wxXML_ELEMENT_NODE ,"building" );
    nodeBuilding->AddAttribute("id", "Rôtillon");
   
    wxXmlNode * nodeType = new wxXmlNode(wxXML_ELEMENT_NODE ,"building_type");
    wxXmlNode * nodeTypeValue = new wxXmlNode(wxXML_TEXT_NODE ,"building_type", "hôpital" );
    nodeType->AddChild ( nodeTypeValue );

    wxXmlNode * nodeLocation = new wxXmlNode(wxXML_ELEMENT_NODE ,"building_location" );
    wxXmlNode * nodeLocationValue = new wxXmlNode(wxXML_TEXT_NODE ,"building_location", "Zürich" );
    nodeLocation->AddChild (nodeLocationValue );

    wxXmlNode * nodeHeight = new wxXmlNode(wxXML_ELEMENT_NODE ,"building_height" );
    wxXmlNode * nodeHeightValue = new wxXmlNode(wxXML_TEXT_NODE ,"building_height", "40" );
    nodeHeight->AddChild ( nodeHeightValue );

    nodeBuilding->AddChild(nodeType);
    nodeBuilding->AddChild(nodeLocation);
    nodeBuilding->AddChild(nodeHeight);
   
    nodeBase->AddChild ( nodeBuilding );

    doc.SetRoot(nodeBase);
    
    doc.Save(filePath);
    
    // Read
    wxXmlDocument doc2;

    bool success = doc2.Load(filePath);
    CHECK_EQUAL(true, success);

    CHECK_EQUAL("base", doc2.GetRoot()->GetName());

    wxXmlNode *childBuilding = doc2.GetRoot()->GetChildren();
    CHECK_EQUAL("Rôtillon", childBuilding->GetAttribute("id"));
    
    wxXmlNode *childBuildingType = childBuilding->GetChildren();
    CHECK_EQUAL("building_type", childBuildingType->GetName());
    CHECK_EQUAL("hôpital", childBuildingType->GetNodeContent());
    
    wxXmlNode *childBuildingLocation = childBuildingType->GetNext();
    CHECK_EQUAL("building_location", childBuildingLocation->GetName());
    CHECK_EQUAL("Zürich", childBuildingLocation->GetNodeContent());
    
    wxXmlNode *childBuildingHeight = childBuildingLocation->GetNext();
    CHECK_EQUAL("building_height", childBuildingHeight->GetName());
    CHECK_EQUAL("40", childBuildingHeight->GetNodeContent());

    asRemoveDir(tmpDir);
}

TEST(SaveAndLoadXmlFileAtmoSwingStyle)
{
    wxString tmpDir = asConfig::CreateTempFileName("xmlFileTest");
    wxFileName::Mkdir(tmpDir);
    wxString filePath = tmpDir + wxFileName::GetPathSeparator() + "file2.xml";

    // Write
    asFileXml fileXml(filePath, asFile::New);

    wxXmlNode * nodeBuilding = new wxXmlNode(wxXML_ELEMENT_NODE ,"building" );
    nodeBuilding->AddAttribute("id", "Rôtillon");
   
    nodeBuilding->AddChild(fileXml.CreateNodeWithValue("building_type", "hôpital"));
    nodeBuilding->AddChild(fileXml.CreateNodeWithValue("building_location", "Zürich"));
    nodeBuilding->AddChild(fileXml.CreateNodeWithValue("building_height", "40"));

    fileXml.AddChild(nodeBuilding);

    fileXml.Save();
    
    // Read
    asFileXml fileXml2(filePath, asFile::ReadOnly);
    bool success = fileXml2.Open();
    CHECK_EQUAL(true, success);

    CHECK_EQUAL("atmoswing_file", fileXml2.GetRoot()->GetName());

    wxXmlNode *childBuilding = fileXml2.GetRoot()->GetChildren();
    CHECK_EQUAL("Rôtillon", childBuilding->GetAttribute("id"));
    
    wxXmlNode *childBuildingType = childBuilding->GetChildren();
    CHECK_EQUAL("building_type", childBuildingType->GetName());
    CHECK_EQUAL("hôpital", childBuildingType->GetNodeContent());
    
    wxXmlNode *childBuildingLocation = childBuildingType->GetNext();
    CHECK_EQUAL("building_location", childBuildingLocation->GetName());
    CHECK_EQUAL("Zürich", childBuildingLocation->GetNodeContent());
    
    wxXmlNode *childBuildingHeight = childBuildingLocation->GetNext();
    CHECK_EQUAL("building_height", childBuildingHeight->GetName());
    CHECK_EQUAL("40", childBuildingHeight->GetNodeContent());

    asRemoveDir(tmpDir);
}
/*
TEST(LoadSimpleXmlFile)
{
    wxString str("Testing xml files...\n");
    printf("%s", str.mb_str(wxConvUTF8).data());

    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/file_xml.xml");

    bool success = asFileXml::Test(filepath);

    
    CHECK_EQUAL(true, success);


}

TEST(LoadSimpleXmlFile)
{
    wxString str("Testing xml files...\n");
    printf("%s", str.mb_str(wxConvUTF8).data());

    wxString filepath = wxFileName::GetCwd();
    filepath.Append("/files/file_xml_error.xml");

    bool success = asFileXml::Test(filepath);

    
    CHECK_EQUAL(false, success);


}
*/
}
