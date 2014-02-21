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
 
#include "asFileXml.h"

const wxChar* XmlNodeSeparator = wxT(".");
const wxChar* XmlIdStart = wxT("[");
const wxChar* XmlIdEnd = wxT("]");

asFileXml::asFileXml(const wxString &FileName, const ListFileMode &FileMode)
:
asFile(FileName, FileMode)
{
    m_BaseNodePointer = 0;
    m_BaseNodeName = wxEmptyString;
    m_Document = NULL;
}

asFileXml::~asFileXml()
{
    wxDELETE(m_Document);
}

bool asFileXml::Open()
{
    if (!Find()) return false;

    try
    {
        std::string stlDocPath = std::string(m_FileName.GetFullPath().mb_str());
        ticpp::Document doc(stlDocPath);

        m_Document = new ticpp::Document(stlDocPath);
        m_ElementPointer = 0;
        m_ElementName = wxEmptyString;
        if ((Exists()) & (m_FileMode!=asFile::Replace))
        {
            m_Document->LoadFile(TIXML_ENCODING_UTF8);
        }
    }
    catch ( ticpp::Exception& ex )
    {
        wxString msg(ex.what(), wxConvUTF8);
        asThrowException(wxString::Format(_("Error in parsing the xml file: %s"), msg.c_str()));
    }

    // If new, set declaration and the root element
    if ( (m_FileMode==asFile::New) | (m_FileMode==asFile::Replace) )
    {
        ticpp::Declaration decl("1.0", "UTF-8", "");
        m_Document->InsertEndChild (decl);

        ticpp::Element root("AtmoSwingFile");
        root.SetAttribute("version", 0.3);
        m_Document->InsertEndChild(root);
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

    m_Document->SaveFile();
    return true;
}

bool asFileXml::UpdateElementName()
{
    wxASSERT(m_Opened);

    if(m_ElementPointer==NULL)
    {
        wxString msg = wxString::Format(_("Trying to access a node that doesn't exist (maybe: '%s' in '%s')"), m_BaseNodeName.AfterLast('.').c_str(), m_BaseNodeName.c_str());
        asLogWarning(msg);
        return false;
    }
    std::string name = m_ElementPointer->Value();
    wxString namewx (name.c_str(), wxConvUTF8);
    m_ElementName = namewx;

    return true;
}

bool asFileXml::GoToFirstNodeWithPath(const wxString node, const int &showWarnings)
{
    wxASSERT(m_Opened);

    if (!m_BaseNodeName.IsEmpty())
    {
        m_BaseNodeName = m_BaseNodeName + "." + node;
    } else {
        m_BaseNodeName = node;
    }
    if (!SetPointerFirstElement(node)) 
    {
        if (showWarnings==asSHOW_WARNINGS)
        {
            asLogError(wxString::Format(_("Cannot find node %s"), node.c_str()));
        }
        return false;
    }

    m_BaseNodePointer = m_ElementPointer;
    if (!UpdateElementName())
    {
        if (showWarnings==asSHOW_WARNINGS)
        {
            asLogError(wxString::Format(_("Cannot go to node %s"), node.c_str()));
        }
        return false;
    }

    return true;
}

bool asFileXml::GoToLastNodeWithPath(const wxString node, const int &showWarnings)
{
    wxASSERT(m_Opened);

    if (!m_BaseNodeName.IsEmpty())
    {
        m_BaseNodeName = m_BaseNodeName + "." + node;
    } else {
        m_BaseNodeName = node;
    }
    if (!SetPointerLastElement(node)) 
    {
        if (showWarnings==asSHOW_WARNINGS)
        {
            asLogError(wxString::Format(_("Cannot find node %s"), node.c_str()));
        }
        return false;
    }

    m_BaseNodePointer = m_ElementPointer;
    if (!UpdateElementName())
    {
        if (showWarnings==asSHOW_WARNINGS)
        {
            asLogError(wxString::Format(_("Cannot go to node %s"), node.c_str()));
        }
        return false;
    }

    return true;
}

bool asFileXml::GoToNextSameNode()
{
    wxASSERT(m_Opened);

    m_ElementPointer = m_BaseNodePointer; // Set the element pointer back to base node
    bool result = SetPointerNextSameSibling(false);
    m_BaseNodePointer = m_ElementPointer;

    return result;
}

bool asFileXml::GoToNextSameNodeWithAttributeValue(const wxString &attributeName, const wxString &attributeValue, const int &showWarnings)
{
    wxASSERT(m_Opened);

    m_ElementPointer = m_BaseNodePointer; // Set the element pointer back to base node

    // Put the attribute name into a standard string
    std::string attributeNameStd = std::string(attributeName.mb_str());

    // Continue as long as the attribute value doesn't match
    while(true)
    {
        if (!SetPointerNextSameSibling(false))
        {
            m_ElementPointer = m_BaseNodePointer; // Restore pointer
            if(showWarnings==asSHOW_WARNINGS) 
            {
                asLogError(wxString::Format(_("The attribute '%s' with value '%s' cannot be found in the xml file for the element '%s'"), attributeName.c_str(), attributeValue.c_str(), m_ElementName.c_str()));
            }
            return false;
        }

        // Check attribute value
        wxString thisAttributeValue(m_ElementPointer->GetAttribute(attributeNameStd).c_str(), wxConvUTF8);;
        if (thisAttributeValue.IsSameAs(attributeValue,false))
        {
            m_BaseNodePointer = m_ElementPointer; // Set base node pointer
            return true;
        }
    }
    
    if(showWarnings==asSHOW_WARNINGS) 
    {
        asLogError(wxString::Format(_("The attribute '%s' with value '%s' cannot be found in the xml file for the element '%s'"), attributeName.c_str(), attributeValue.c_str(), m_ElementName.c_str()));
    }

    return false;
}

bool asFileXml::GoANodeBack()
{
    wxASSERT(m_Opened);

    m_BaseNodePointer = (ticpp::Element*)m_BaseNodePointer->Parent();
    m_BaseNodeName = m_BaseNodeName.BeforeLast('.');
    m_ElementPointer = m_BaseNodePointer;
    if (!UpdateElementName()) return false;

    return true;
}

bool asFileXml::GoToChildNodeWithAttributeValue(const wxString &attributeName, const wxString &attributeValue, const int &showWarnings)
{
    wxASSERT(m_Opened);

    ticpp::Element* elementPointerBkp = m_ElementPointer;

    m_ElementPointer = GetChildFromAttributeValue(m_ElementPointer, attributeName, attributeValue);
    if (m_ElementPointer==NULL)
    {
        m_ElementPointer = elementPointerBkp;
        if(showWarnings==asSHOW_WARNINGS) 
        {
            asLogError(wxString::Format(_("The attribute '%s' with value '%s' cannot be found in the xml file for the element '%s'"), attributeName.c_str(), attributeValue.c_str(), m_ElementName.c_str()));
        }
        return false;
    }
    m_BaseNodePointer = m_ElementPointer;
    std::string name = m_ElementPointer->Value();
    wxString nodeName (name.c_str(), wxConvUTF8);
    m_BaseNodeName.Append(".");
    m_BaseNodeName.Append(nodeName);

    if (!UpdateElementName())
    {
        if(showWarnings==asSHOW_WARNINGS) 
        {
            asLogError(wxString::Format(_("The attribute '%s' with value '%s' cannot be found in the xml file for the element '%s'"), attributeName.c_str(), attributeValue.c_str(), m_ElementName.c_str()));
        }
        return false;
    }

    return true;
}

bool asFileXml::GoToLastChildNodeWithAttributeValue(const wxString &attributeName, const wxString &attributeValue, const int &showWarnings)
{
    wxASSERT(m_Opened);

    ticpp::Element* elementPointerBkp = m_ElementPointer;

    m_ElementPointer = GetChildFromAttributeValue(m_ElementPointer, attributeName, attributeValue);
    if (m_ElementPointer==NULL)
    {
        m_ElementPointer = elementPointerBkp;
        if(showWarnings==asSHOW_WARNINGS) 
        {
            asLogError(wxString::Format(_("The attribute '%s' with value '%s' cannot be found in the xml file for the element '%s'"), attributeName.c_str(), attributeValue.c_str(), m_ElementName.c_str()));
        }
        return false;
    }

    m_BaseNodePointer = m_ElementPointer;
    std::string name = m_ElementPointer->Value();
    wxString nodeName (name.c_str(), wxConvUTF8);
    m_BaseNodeName.Append(".");
    m_BaseNodeName.Append(nodeName);

    if (!UpdateElementName())
    {
        if(showWarnings==asSHOW_WARNINGS) 
        {
            asLogError(wxString::Format(_("The attribute '%s' with value '%s' cannot be found in the xml file for the element '%s'"), attributeName.c_str(), attributeValue.c_str(), m_ElementName.c_str()));
        }
        return false;
    }

    // Go to next same node
    bool doContinue = true;
    while (doContinue)
    {
        doContinue = GoToNextSameNodeWithAttributeValue(attributeName, attributeValue, asHIDE_WARNINGS);
    }

    return true;
}

wxString asFileXml::GetCurrenNodePath()
{
    wxASSERT(m_Opened);

    return m_BaseNodeName;
}

void asFileXml::ClearCurrenNodePath()
{
    wxASSERT(m_Opened);

    m_BaseNodeName = wxEmptyString;
    m_BaseNodePointer = 0;
}

bool asFileXml::SetPointerFirstElement(const wxString &elementAccessConst)
{
    wxASSERT(m_Opened);

    vector< wxString > VElements;
    wxString ElementAccess = elementAccessConst;
    wxString ThisElement, ElementName, ElementIdValue;
    std::string stlThisElement;

    try
    {
        while (!ElementAccess.IsNull())
        {
            VElements.push_back(ElementAccess.BeforeFirst(*XmlNodeSeparator));
            ElementAccess = ElementAccess.AfterFirst(*XmlNodeSeparator);
        }

        // Get the root node
        int nodestart = 0;
        if (m_BaseNodePointer)
        {
            m_ElementPointer = m_BaseNodePointer;
            nodestart = 0;
        } else {
            m_ElementPointer = m_Document->FirstChildElement(VElements[0].mb_str(wxConvUTF8));
            nodestart = 1;
        }

        // Set the pointer on the correct node
        for(size_t i_node=nodestart; i_node<VElements.size(); i_node++)
        {
            ThisElement = VElements[i_node];

            // Get the markup name without any id tag
            ElementName = ThisElement.BeforeFirst(*XmlIdStart);

            // Check if the node exists. Only the last one is allowed to be vacant.
            if ((i_node!=VElements.size()-1) && (!MarkupExists(m_ElementPointer, ElementName)))
            {
                asLogWarning(wxString::Format(_("The xml file does not contain the following markup: '%s'"), ElementName.c_str()));
                return false;
            }
            else if ((i_node==VElements.size()-1) && (!MarkupExists(m_ElementPointer, ElementName)))
            {
                // If last element not found, set null
                m_ElementPointer = NULL;
            }
            else
            {
                // Check if an Id is given for this element
                ElementIdValue = ThisElement.AfterFirst(*XmlIdStart);

                if (!ElementIdValue.IsNull())
                {
                    ElementIdValue = ElementIdValue.BeforeLast(*XmlIdEnd);
                    m_ElementPointer = GetChildFromAttributeValue(m_ElementPointer, "id", ElementIdValue);
                    if (m_ElementPointer==NULL)
                    {
                        asLogWarning(wxString::Format(_("The id '%s' cannot be found in the xml file for the element '%s'"), ElementIdValue.c_str(), ElementName.c_str()));
                        return false;
                    }
                    m_ElementName = ThisElement;

                } else {
                    stlThisElement = std::string(ThisElement.mb_str());
                    m_ElementPointer = m_ElementPointer->FirstChildElement(stlThisElement);
                    m_ElementName = ThisElement;
                }
            }
        }
    }
    catch ( ticpp::Exception& ex )
    {
        wxString msg(ex.what(), wxConvUTF8);
        asLogWarning(wxString::Format(_("Error in parsing the xml file: %s"), msg.c_str()));
        return false;
    }

    return true;
}

bool asFileXml::SetPointerLastElement(const wxString &elementAccessConst)
{
    wxASSERT(m_Opened);

    if(!SetPointerFirstElement(elementAccessConst)) return false;

    // Find the last sibling
    while(SetPointerNextSameSibling(false))
    {
        // Nothing to do
    }

    return true;
}

bool asFileXml::SetPointerNextSameSibling(const bool &throwIfNoSiblings)
{
    wxASSERT(m_Opened);

    ticpp::Element* pElementTmp;

    if (!UpdateElementName()) return false;
    if (m_ElementName.length()==0) asThrowException(_("There was no previous element. Please choose an element first before looking for siblings."));

    pElementTmp = m_ElementPointer->NextSiblingElement (m_ElementName.mb_str(), throwIfNoSiblings);
    if (pElementTmp)
    {
        m_ElementPointer = pElementTmp;
    }
    else
    {
        return false;
    }
    return true;
}

bool asFileXml::GetNextElement(const wxString &elementAccess)
{
    wxASSERT(m_Opened);

    wxString ElementAccessStr = elementAccess;
    if (!ElementAccessStr.IsSameAs(ElementAccessStr.AfterLast(*XmlNodeSeparator)))
    {
        ElementAccessStr = ElementAccessStr.AfterLast(*XmlNodeSeparator);
    }
    wxString ElementMember = m_ElementName;

    if (!ElementMember.IsSameAs(ElementAccessStr))
    {
        wxString msg = wxString::Format(_("The given element name ('%s') does not match the previous one ('%s')."), ElementAccessStr.c_str(),  ElementMember.c_str());
        asThrowException( msg );
    }
    try
    {
        if (!SetPointerNextSameSibling()) return false;
    }
    catch ( ticpp::Exception& ex )
    {
        wxString msg(ex.what(), wxConvUTF8);
        asThrowException(wxString::Format(_("Error in parsing the xml file: %s"), msg.c_str()));
    }

    return true;
}

template< class T >
T asFileXml::GetFirstElementValue(const wxString &elementAccess, const T &valDefault)
{
    wxASSERT(m_Opened);

    T valreturn;
    try
    {
        if (!SetPointerFirstElement(elementAccess)) return valDefault;
        if (m_ElementPointer==NULL) return valDefault;

        // Get the text value
        m_ElementPointer->GetTextOrDefault(&valreturn,valDefault);
    }
    catch ( ticpp::Exception& ex )
    {
        wxString msg(ex.what(), wxConvUTF8);
        asThrowException(wxString::Format(_("Error in parsing the xml file: %s"), msg.c_str()));
    }

    return valreturn;
}

template< class T >
T asFileXml::GetThisElementValue(const T &ValDefault)
{
    wxASSERT(m_Opened);

    T valreturn;

    try
    {
        if (m_ElementPointer==NULL) return ValDefault;

        // Get the text value
        m_ElementPointer->GetTextOrDefault(&valreturn, ValDefault);
    }
    catch ( ticpp::Exception& ex )
    {
        wxString msg(ex.what(), wxConvUTF8);
        asThrowException(wxString::Format(_("Error in parsing the xml file: %s"), msg.c_str()));
    }

    return valreturn;
}

wxString asFileXml::GetFirstElementValueText(const wxString &ElementAccess, const wxString &ValDefault)
{
    wxASSERT(m_Opened);

    std::string stlRet;
    std::string stlDef = std::string(ValDefault.mb_str());
    stlRet = asFileXml::GetFirstElementValue<std::string>(ElementAccess, stlDef);
    wxString wxRet(stlRet.c_str(), wxConvUTF8);
    return wxRet;
}

int asFileXml::GetFirstElementValueInt(const wxString &ElementAccess, const int &ValDefault)
{
    wxASSERT(m_Opened);

    return asFileXml::GetFirstElementValue<int>(ElementAccess, ValDefault);
}

float asFileXml::GetFirstElementValueFloat(const wxString &ElementAccess, const float &ValDefault)
{
    wxASSERT(m_Opened);

    return asFileXml::GetFirstElementValue<float>(ElementAccess, ValDefault);
}

double asFileXml::GetFirstElementValueDouble(const wxString &ElementAccess, const double &ValDefault)
{
    wxASSERT(m_Opened);

    return asFileXml::GetFirstElementValue<double>(ElementAccess, ValDefault);
}

bool asFileXml::GetFirstElementValueBool(const wxString &ElementAccess, const bool &ValDefault)
{
    wxASSERT(m_Opened);

    return asFileXml::GetFirstElementValue<bool>(ElementAccess, ValDefault);
}

wxString asFileXml::GetThisElementValueText(const wxString &ValDefault)
{
    wxASSERT(m_Opened);

    std::string stlRet;
    std::string stlDef = std::string(ValDefault.mb_str());
    stlRet = asFileXml::GetThisElementValue<std::string>(stlDef);
    wxString wxRet(stlRet.c_str(), wxConvUTF8);
    return wxRet;
}

int asFileXml::GetThisElementValueInt(const int &ValDefault)
{
    wxASSERT(m_Opened);

    return asFileXml::GetThisElementValue<int>(ValDefault);
}

float asFileXml::GetThisElementValueFloat(const float &ValDefault)
{
    wxASSERT(m_Opened);

    return asFileXml::GetThisElementValue<float>(ValDefault);
}

double asFileXml::GetThisElementValueDouble(const double &ValDefault)
{
    wxASSERT(m_Opened);

    return asFileXml::GetThisElementValue<double>(ValDefault);
}

bool asFileXml::GetThisElementValueBool(const bool &ValDefault)
{
    wxASSERT(m_Opened);

    return asFileXml::GetThisElementValue<bool>(ValDefault);
}

template< class T >
T asFileXml::GetFirstElementAttributeValue(const wxString &ElementAccess, const wxString &AttributeName, const T &ValDefault)
{
    wxASSERT(m_Opened);

    T valreturn;
    std::string AttributeNameStr = std::string(AttributeName.mb_str());

    try
    {
        if (!SetPointerFirstElement(ElementAccess)) return ValDefault;
        if (m_ElementPointer==NULL) return ValDefault;

        // Get the attribute value
        m_ElementPointer->GetAttributeOrDefault(AttributeNameStr, &valreturn, ValDefault);

    }
    catch ( ticpp::Exception& ex )
    {
        wxString msg(ex.what(), wxConvUTF8);
        asThrowException(wxString::Format(_("Error in parsing the xml file (searching for '%s' with attribute '%s'): %s"), ElementAccess.c_str(), AttributeName.c_str(), msg.c_str()));
    }

    return valreturn;
}

template< class T >
T asFileXml::GetThisElementAttributeValue(const wxString &AttributeName, const T &ValDefault)
{
    wxASSERT(m_Opened);

    T valreturn;
    std::string AttributeNameStr = std::string(AttributeName.mb_str());

    try
    {
        if (m_ElementPointer==NULL) return ValDefault;

        // Get the attribute value
        m_ElementPointer->GetAttributeOrDefault(AttributeNameStr, &valreturn, ValDefault);
    }
    catch ( ticpp::Exception& ex )
    {
        wxString msg(ex.what(), wxConvUTF8);
        asThrowException(wxString::Format(_("Error in parsing the xml file: %s"), msg.c_str()));
    }

    return valreturn;
}

wxString asFileXml::GetFirstElementAttributeValueText(const wxString &ElementAccess, const wxString &AttributeName, const wxString &ValDefault)
{
    wxASSERT(m_Opened);

    std::string stlRet;
    std::string stlDef = std::string(ValDefault.mb_str());
    stlRet = asFileXml::GetFirstElementAttributeValue<std::string>(ElementAccess, AttributeName, stlDef);
    wxString wxRet(stlRet.c_str(), wxConvUTF8);
    return wxRet;
}

int asFileXml::GetFirstElementAttributeValueInt(const wxString &ElementAccess, const wxString &AttributeName, const int &ValDefault)
{
    wxASSERT(m_Opened);

    return asFileXml::GetFirstElementAttributeValue<int>(ElementAccess, AttributeName, ValDefault);
}

float asFileXml::GetFirstElementAttributeValueFloat(const wxString &ElementAccess, const wxString &AttributeName, const float &ValDefault)
{
    wxASSERT(m_Opened);

    return asFileXml::GetFirstElementAttributeValue<float>(ElementAccess, AttributeName, ValDefault);
}

double asFileXml::GetFirstElementAttributeValueDouble(const wxString &ElementAccess, const wxString &AttributeName, const double &ValDefault)
{
    wxASSERT(m_Opened);

    return asFileXml::GetFirstElementAttributeValue<double>(ElementAccess, AttributeName, ValDefault);
}

bool asFileXml::GetFirstElementAttributeValueBool(const wxString &ElementAccess, const wxString &AttributeName, const bool &ValDefault)
{
    wxASSERT(m_Opened);

    return asFileXml::GetFirstElementAttributeValue<bool>(ElementAccess, AttributeName, ValDefault);
}

wxString asFileXml::GetThisElementAttributeValueText(const wxString &AttributeName, const wxString &ValDefault)
{
    wxASSERT(m_Opened);

    std::string stlRet;
    std::string stlDef = std::string(ValDefault.mb_str());
    stlRet = asFileXml::GetThisElementAttributeValue<std::string>(AttributeName, stlDef);
    wxString wxRet(stlRet.c_str(), wxConvUTF8);
    return wxRet;
}

int asFileXml::GetThisElementAttributeValueInt(const wxString &AttributeName, const int &ValDefault)
{
    wxASSERT(m_Opened);

    return asFileXml::GetThisElementAttributeValue<int>(AttributeName, ValDefault);
}

float asFileXml::GetThisElementAttributeValueFloat(const wxString &AttributeName, const float &ValDefault)
{
    wxASSERT(m_Opened);

    return asFileXml::GetThisElementAttributeValue<float>(AttributeName, ValDefault);
}

double asFileXml::GetThisElementAttributeValueDouble(const wxString &AttributeName, const double &ValDefault)
{
    wxASSERT(m_Opened);

    return asFileXml::GetThisElementAttributeValue<double>(AttributeName, ValDefault);
}

bool asFileXml::GetThisElementAttributeValueBool(const wxString &AttributeName, const bool &ValDefault)
{
    wxASSERT(m_Opened);

    return asFileXml::GetThisElementAttributeValue<bool>(AttributeName, ValDefault);
}

bool asFileXml::MarkupExists(const ticpp::Element *pElem, const wxString &MarkupName)
{
    wxASSERT(m_Opened);

    try
    {
        if (FindMarkup(pElem, MarkupName)==NULL) return false;
        else return true;
    }
    catch ( ticpp::Exception& ex )
    {
        wxString msg(ex.what(), wxConvUTF8);
        asThrowException(wxString::Format(_("Error in parsing the xml file: %s"), msg.c_str()));
    }
}

ticpp::Element* asFileXml::FindMarkup(const ticpp::Element *pElem, const wxString &MarkupName)
{
    wxASSERT(m_Opened);

    try
    {
        ticpp::Iterator<ticpp::Element> child;
        for (child=child.begin(pElem);child!=child.end();child++)
        {
            if (child.Get()->Value().compare(MarkupName.mb_str())==0)return child.Get();
        }
    }
    catch ( ticpp::Exception& ex )
    {
        wxString msg(ex.what(), wxConvUTF8);
        asThrowException(wxString::Format(_("Error in parsing the xml file: %s"), msg.c_str()));
    }
    return NULL;
}

ticpp::Element* asFileXml::GetChildFromAttributeValue(const ticpp::Element* pElem, const wxString &AttributeName, const wxString &AttributeValue)
{
    wxASSERT(m_Opened);

    try
    {
        ticpp::Iterator<ticpp::Element> child;
        for (child=child.begin(pElem);child!=child.end();child++)
        {
            // Put the AttributeName into a standard string
            std::string AttributeNameStd = std::string(AttributeName.mb_str());
            wxString thisAttributeValue(child.Get()->GetAttribute(AttributeNameStd).c_str(), wxConvUTF8);;
            if (thisAttributeValue.IsSameAs(AttributeValue,false))
            {
                return child.Get();
            }
        }
    }
    catch ( ticpp::Exception& ex )
    {
        wxString msg(ex.what(), wxConvUTF8);
        asThrowException(wxString::Format(_("Error in parsing the xml file: %s"), msg.c_str()));
    }
    return NULL;
}

bool asFileXml::InsertElement(const wxString &ParentNodeName, const wxString &NewNodeName)
{
    wxASSERT(m_Opened);

    // Put the data into a standard string
    std::string NewNodeNameStd = std::string(NewNodeName.mb_str());
    if (!ParentNodeName.IsEmpty())
    {
        if (!SetPointerFirstElement(ParentNodeName)) return false;
    }
    ticpp::Element newnode(NewNodeNameStd);
    m_ElementPointer->InsertEndChild (newnode);

    return true;
}

bool asFileXml::InsertElementAndValue(const wxString &ParentNodeName, const wxString &NewNodeName, const wxString &Value)
{
    wxASSERT(m_Opened);

    // Put the data into a standard string
    std::string NewNodeNameStd = std::string(NewNodeName.mb_str());
    std::string ValueStd = std::string(Value.mb_str());
    if (!ParentNodeName.IsEmpty())
    {
        if (!SetPointerFirstElement(ParentNodeName)) return false;
    }
    ticpp::Element newnode(NewNodeNameStd);
    newnode.SetText(ValueStd);
    m_ElementPointer->InsertEndChild (newnode);

    return true;
}

bool asFileXml::InsertElementAndAttribute(const wxString &ParentNodeName, const wxString &NewNodeName, const wxString &AttributeName, const wxString &AttributeValue)
{
    wxASSERT(m_Opened);

    // Put the data into a standard string
    std::string NewNodeNameStd = std::string(NewNodeName.mb_str());
    std::string AttributeNameStd = std::string(AttributeName.mb_str());
    std::string AttributeValueStd = std::string(AttributeValue.mb_str());
    if (!ParentNodeName.IsEmpty())
    {
        if (!SetPointerFirstElement(ParentNodeName)) return false;
    }
    ticpp::Element newnode(NewNodeNameStd);
    newnode.SetAttribute(AttributeNameStd, AttributeValueStd);
    m_ElementPointer->InsertEndChild (newnode);

    return true;
}

bool asFileXml::SetElementAttribute(const wxString &ParentNodeName, const wxString &AttributeName, const wxString &AttributeValue)
{
    wxASSERT(m_Opened);

    // Put the data into a standard string
    std::string AttributeNameStd = std::string(AttributeName.mb_str());
    std::string AttributeValueStd = std::string(AttributeValue.mb_str());

    if (!ParentNodeName.IsEmpty())
    {
        if (!SetPointerFirstElement(ParentNodeName)) return false;
    }
    m_ElementPointer->SetAttribute(AttributeNameStd, AttributeValueStd);

    return true;
}

bool asFileXml::SetElementValue(const wxString &ParentNodeName, const wxString &Value)
{
    wxASSERT(m_Opened);

    // Put the data into a standard string
    std::string ValueStd = std::string(Value.mb_str());
    if (!ParentNodeName.IsEmpty())
    {
        if (!SetPointerFirstElement(ParentNodeName)) return false;
    }
    m_ElementPointer->SetText(ValueStd);

    return true;
}
