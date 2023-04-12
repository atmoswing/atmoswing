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

#include "asFileXml.h"

asFileXml::asFileXml(const wxString& fileName, const FileMode& fileMode)
    : asFile(fileName, fileMode) {}

bool asFileXml::Open() {
    if (!Find()) return false;

    if ((Exists()) & (m_fileMode != asFile::Replace)) {
        if (!m_document.Load(m_fileName.GetFullPath())) {
            wxLogError(_("Couldn't open the xml file %s"), m_fileName.GetFullPath());
            return false;
        }
    }

    // If new, set declaration and the root element
    if ((m_fileMode == asFile::New) | (m_fileMode == asFile::Replace)) {
        wxXmlNode* nodeBase = new wxXmlNode(wxXML_ELEMENT_NODE, "atmoswing");
        nodeBase->AddAttribute("version", "1.0");  // AtmoSwing file version
        m_document.SetRoot(nodeBase);
    }

    m_opened = true;

    return true;
}

bool asFileXml::Close() {
    wxASSERT(m_opened);

    return true;
}

bool asFileXml::Save() {
    wxASSERT(m_opened);

    m_document.Save(m_fileName.GetFullPath());
    return true;
}

void asFileXml::AddChild(wxXmlNode* node) {
    GetRoot()->AddChild(node);
}

bool asFileXml::CheckRootElement() const {
    if (!GetRoot()) return false;
    if (!IsAnAtmoSwingFile()) return false;

    return FileVersionIsOrAbove(1.0);
}

wxXmlNode* asFileXml::CreateNodeWithValue(const wxString& name, const bool& content) {
    wxString value;
    value << content;

    return CreateNodeWithValue(name, value);
}

wxXmlNode* asFileXml::CreateNodeWithValue(const wxString& name, const int& content) {
    wxString value;
    value << content;

    return CreateNodeWithValue(name, value);
}

wxXmlNode* asFileXml::CreateNodeWithValue(const wxString& name, const float& content) {
    wxString value;
    value << content;

    return CreateNodeWithValue(name, value);
}

wxXmlNode* asFileXml::CreateNodeWithValue(const wxString& name, const double& content) {
    wxString value;
    value << content;

    return CreateNodeWithValue(name, value);
}

wxXmlNode* asFileXml::CreateNodeWithValue(const wxString& name, const wxString& content) {
    auto node = new wxXmlNode(wxXML_ELEMENT_NODE, name);
    auto nodeValue = new wxXmlNode(wxXML_TEXT_NODE, name, content);
    node->AddChild(nodeValue);

    return node;
}

bool asFileXml::IsAnAtmoSwingFile() const {
    if (!GetRoot()) return false;
    if (GetRoot()->GetName().IsSameAs("AtmoSwingFile", false)) {
        wxLogError(_("The file %s is for an old version of AtmoSwing and is no longer supported (root: %s)."),
                   m_fileName.GetFullName(), GetRoot()->GetName());
        return false;
    }
    if (!GetRoot()->GetName().IsSameAs("atmoswing", false)) {
        wxLogError(_("The file %s is not an AtmoSwing file (root: %s)."), m_fileName.GetFullName(),
                   GetRoot()->GetName());
        return false;
    }
    return true;
}

bool asFileXml::FileVersionIsOrAbove(const float version) const {
    if (!GetRoot()) return false;
    wxString fileVersionStr = GetRoot()->GetAttribute("version");
    double fileVersion;

    if (!fileVersionStr.ToDouble(&fileVersion) || (float)fileVersion < version) {
        wxLogError(_("The file version of %s is no longer supported."), m_fileName.GetFullName());
        return false;
    }
    return true;
}

void asFileXml::UnknownNode(wxXmlNode* node) {
    wxLogVerbose(_("An unknown element was found in the file: %s"), node->GetName());
}

bool asFileXml::GetBool(wxXmlNode* node, const bool defaultValue) {
    if (!node->GetChildren()) {
        wxLogWarning(_("The node is empty in the xml file."));
        return defaultValue;
    }

    wxString valueStr = node->GetChildren()->GetContent();
    if (valueStr.IsSameAs("true", false)) {
        return true;
    } else if (valueStr.IsSameAs("false", false)) {
        return false;
    } else if (valueStr.IsSameAs("T", false)) {
        return true;
    } else if (valueStr.IsSameAs("F", false)) {
        return false;
    } else if (valueStr.IsSameAs("1", false)) {
        return true;
    } else if (valueStr.IsSameAs("0", false)) {
        return false;
    } else if (valueStr.IsSameAs("yes", false)) {
        return true;
    } else if (valueStr.IsSameAs("no", false)) {
        return false;
    } else if (valueStr.IsSameAs("y", false)) {
        return true;
    } else if (valueStr.IsSameAs("n", false)) {
        return false;
    } else if (valueStr.IsEmpty()) {
        return defaultValue;
    } else {
        wxLogError(_("Failed at converting the value of the element %s (XML file)."), node->GetName());
    }

    return false;
}

int asFileXml::GetInt(wxXmlNode* node, const int defaultValue) {
    if (!node->GetChildren()) {
        wxLogWarning(_("The node is empty in the xml file."));
        return defaultValue;
    }

    long value;
    wxString valueStr = node->GetChildren()->GetContent();
    if (valueStr.IsEmpty()) {
        return defaultValue;
    }
    if (!valueStr.ToLong(&value)) {
        wxLogError(_("Failed at converting the value of the element %s (XML file)."), node->GetName());
    }
    return (int)value;
}

float asFileXml::GetFloat(wxXmlNode* node, const float defaultValue) {
    if (!node->GetChildren()) {
        wxLogWarning(_("The node is empty in the xml file."));
        return defaultValue;
    }

    double value;
    wxString valueStr = node->GetChildren()->GetContent();
    if (valueStr.IsEmpty()) {
        return defaultValue;
    }
    if (!valueStr.ToDouble(&value)) {
        wxLogError(_("Failed at converting the value of the element %s (XML file)."), node->GetName());
    }
    return (float)value;
}

double asFileXml::GetDouble(wxXmlNode* node, const double defaultValue) {
    if (!node->GetChildren()) {
        wxLogWarning(_("The node is empty in the xml file."));
        return defaultValue;
    }

    double value;
    wxString valueStr = node->GetChildren()->GetContent();
    if (valueStr.IsEmpty()) {
        return defaultValue;
    }
    if (!valueStr.ToDouble(&value)) {
        wxLogError(_("Failed at converting the value of the element %s (XML file)."), node->GetName());
    }
    return value;
}

wxString asFileXml::GetString(wxXmlNode* node, const wxString& defaultValue) {
    if (!node->GetChildren()) {
        wxLogWarning(_("The node is empty in the xml file."));
        return wxEmptyString;
    }

    wxString value = node->GetChildren()->GetContent();
    if (value.IsEmpty()) {
        return defaultValue;
    }
    return value;
}

bool asFileXml::GetAttributeBool(wxXmlNode* node, const wxString& attribute, bool defaultValue, bool raiseWarning) {
    wxString attrVal = node->GetAttribute(attribute);
    if (attrVal.IsSameAs("true", false)) {
        return true;
    } else if (attrVal.IsSameAs("false", false)) {
        return false;
    } else if (attrVal.IsSameAs("T", false)) {
        return true;
    } else if (attrVal.IsSameAs("F", false)) {
        return false;
    } else if (attrVal.IsSameAs("1", false)) {
        return true;
    } else if (attrVal.IsSameAs("0", false)) {
        return false;
    } else if (attrVal.IsSameAs("yes", false)) {
        return true;
    } else if (attrVal.IsSameAs("no", false)) {
        return false;
    } else if (attrVal.IsSameAs("y", false)) {
        return true;
    } else if (attrVal.IsSameAs("n", false)) {
        return false;
    } else {
        if (raiseWarning) {
            wxLogError(_("Failed at converting the value of the element %s (XML file)."), node->GetName());
        }
    }

    return defaultValue;
}

int asFileXml::GetAttributeInt(wxXmlNode* node, const wxString& attribute) {
    wxString attrVal = node->GetAttribute(attribute);

    if (attrVal.IsEmpty()) {
        wxLogError(_("Empty %s attribute of the element %s (XML file)."), attribute, node->GetName());
        return 0;
    }

    long value;
    if (!attrVal.ToLong(&value)) {
        wxLogError(_("Failed at converting the value of the element %s (XML file)."), node->GetName());
    }
    return (int)value;
}

float asFileXml::GetAttributeFloat(wxXmlNode* node, const wxString& attribute) {
    wxString attrVal = node->GetAttribute(attribute);

    if (attrVal.IsEmpty()) {
        wxLogError(_("Empty %s attribute of the element %s (XML file)."), attribute, node->GetName());
        return NAN;
    }

    double value;
    if (!attrVal.ToDouble(&value)) {
        wxLogError(_("Failed at converting the value of the element %s (XML file)."), node->GetName());
    }
    return (float)value;
}

double asFileXml::GetAttributeDouble(wxXmlNode* node, const wxString& attribute) {
    wxString attrVal = node->GetAttribute(attribute);

    if (attrVal.IsEmpty()) {
        wxLogError(_("Empty %s attribute of the element %s (XML file)."), attribute, node->GetName());
        return NAN;
    }

    double value;
    if (!attrVal.ToDouble(&value)) {
        wxLogError(_("Failed at converting the value of the element %s (XML file)."), node->GetName());
    }
    return value;
}

wxString asFileXml::GetAttributeString(wxXmlNode* node, const wxString& attribute) {
    wxString attrVal = node->GetAttribute(attribute);

    if (attrVal.IsEmpty()) {
        wxLogError(_("Empty %s attribute of the element %s (XML file)."), attribute, node->GetName());
        return wxEmptyString;
    }
    return attrVal;
}