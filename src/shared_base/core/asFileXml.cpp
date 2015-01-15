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
 * Portions Copyright 2015 Pascal Horton, Terr@num.
 */
 
#include "asFileXml.h"


asFileXml::asFileXml(const wxString &FileName, const ListFileMode &FileMode)
:
asFile(FileName, FileMode)
{
    
}

asFileXml::~asFileXml()
{
    
}

bool asFileXml::Open()
{
    if (!Find()) return false;

    if ((Exists()) & (m_FileMode!=asFile::Replace))
    {
        if(!m_Document.Load(m_FileName.GetFullPath())) {
            asLogError(wxString::Format(_("Couldn't open the xml file %s"), m_FileName.GetFullPath().c_str()));
            return false;
        }
    }

    // If new, set declaration and the root element
    if ( (m_FileMode==asFile::New) | (m_FileMode==asFile::Replace) )
    {
        wxXmlNode * nodeBase = new wxXmlNode(wxXML_ELEMENT_NODE ,"atmoswing_file");
        nodeBase->AddAttribute("version", "1.0"); // AtmoSwing file version
        m_Document.SetRoot(nodeBase);
    }

    m_Opened = true;

    return true;
}

bool asFileXml::Close()
{
    wxASSERT(m_Opened);

    return true;
}

bool asFileXml::Save()
{
    wxASSERT(m_Opened);

    m_Document.Save(m_FileName.GetFullPath());
    return true;
}

void asFileXml::AddChild(wxXmlNode* node)
{
    GetRoot()->AddChild(node);
}

bool asFileXml::CheckRootElement()
{
    if (!GetRoot()) return false;
    if (!IsAnAtmoSwingFile()) return false;
    if (!FileVersionIsOrAbove(1.0)) return false;

    return true;
}

wxXmlNode * asFileXml::CreateNodeWithValue(const wxString &name, const bool &content)
{
    wxString value;
    value << content;

    return CreateNodeWithValue(name, value);
}

wxXmlNode * asFileXml::CreateNodeWithValue(const wxString &name, const int &content)
{
    wxString value;
    value << content;

    return CreateNodeWithValue(name, value);
}

wxXmlNode * asFileXml::CreateNodeWithValue(const wxString &name, const float &content)
{
    wxString value;
    value << content;

    return CreateNodeWithValue(name, value);
}

wxXmlNode * asFileXml::CreateNodeWithValue(const wxString &name, const double &content)
{
    wxString value;
    value << content;

    return CreateNodeWithValue(name, value);
}

wxXmlNode * asFileXml::CreateNodeWithValue(const wxString &name, const wxString &content)
{
    wxXmlNode * node = new wxXmlNode(wxXML_ELEMENT_NODE, name );
    wxXmlNode * nodeValue = new wxXmlNode(wxXML_TEXT_NODE, name, content );
    node->AddChild (nodeValue );

    return node;
}

bool asFileXml::IsAnAtmoSwingFile()
{
    if (!GetRoot()) return false;
    if (GetRoot()->GetName().IsSameAs("AtmoSwingFile", false))
    {
        asLogError(wxString::Format(_("The file %s is for an old version of AtmoSwing and is no longer supported (root: %s)."), m_FileName.GetFullName(), GetRoot()->GetName()));
        return false;
    }
    if (!GetRoot()->GetName().IsSameAs("atmoswing_file", false))
    {
        asLogError(wxString::Format(_("The file %s is not an AtmoSwing file (root: %s)."), m_FileName.GetFullName(), GetRoot()->GetName()));
        return false;
    }
    return true;
}

bool asFileXml::FileVersionIsOrAbove(const float version)
{
    if (!GetRoot()) return false;
    wxString fileVersionStr = GetRoot()->GetAttribute("version");
    double fileVersion;

    if(!fileVersionStr.ToDouble(&fileVersion) || (float)fileVersion<version)
    {
        asLogError(wxString::Format(_("The file version of %s is no longer supported."), m_FileName.GetFullName()));
        return false;
    }
    return true;
}

void asFileXml::UnknownNode(wxXmlNode *node)
{
    asLogError(wxString::Format(_("An unknown element was found in the file: %s"), node->GetName().c_str()));
}

bool asFileXml::GetBool(wxXmlNode *node, const bool defaultValue)
{
    wxString valueStr = node->GetContent();
    if (valueStr.IsSameAs("true", false)) {
        return true;
    }
    else if (valueStr.IsSameAs("false", false)) {
        return false;
    }
    else if (valueStr.IsSameAs("T", false)) {
        return true;
    }
    else if (valueStr.IsSameAs("F", false)) {
        return false;
    }
    else if (valueStr.IsSameAs("1", false)) {
        return true;
    }
    else if (valueStr.IsSameAs("0", false)) {
        return false;
    }
    else if (valueStr.IsSameAs("yes", false)) {
        return true;
    }
    else if (valueStr.IsSameAs("no", false)) {
        return false;
    }
    else if (valueStr.IsSameAs("y", false)) {
        return true;
    }
    else if (valueStr.IsSameAs("n", false)) {
        return false;
    }
    else if (valueStr.IsEmpty()) {
        return defaultValue;
    }
    else
    {
        asLogError(wxString::Format(_("Failed at converting the value of the element %s (XML file)."), node->GetName().c_str()));
    }

    return false;
}

int asFileXml::GetInt(wxXmlNode *node, const int defaultValue)
{
    long value;
    wxString valueStr = node->GetContent();
    if (valueStr.IsEmpty()) {
        return defaultValue;
    }
    if(!valueStr.ToLong(&value)) {
        asLogError(wxString::Format(_("Failed at converting the value of the element %s (XML file)."), node->GetName().c_str()));
    }
    return (int)value;
}

float asFileXml::GetFloat(wxXmlNode *node, const float defaultValue)
{
    double value;
    wxString valueStr = node->GetContent();
    if (valueStr.IsEmpty()) {
        return defaultValue;
    }
    if(!valueStr.ToDouble(&value)) {
        asLogError(wxString::Format(_("Failed at converting the value of the element %s (XML file)."), node->GetName().c_str()));
    }
    return (float)value;
}

double asFileXml::GetDouble(wxXmlNode *node, const double defaultValue)
{
    double value;
    wxString valueStr = node->GetContent();
    if (valueStr.IsEmpty()) {
        return defaultValue;
    }
    if(!valueStr.ToDouble(&value)) {
        asLogError(wxString::Format(_("Failed at converting the value of the element %s (XML file)."), node->GetName().c_str()));
    }
    return value;
}

wxString asFileXml::GetString(wxXmlNode *node, const wxString &defaultValue)
{
    wxString value = node->GetContent();
    if (value.IsEmpty()) {
        return defaultValue;
    }
    return value;
}