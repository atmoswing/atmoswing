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
 
#ifndef ASFILEXML_H
#define ASFILEXML_H

#include "asIncludes.h"
#include <asFile.h>

#ifndef TIXML_USE_TICPP
#define TIXML_USE_TICPP
#endif
/*
#ifndef TIXML_USE_STL
#define TIXML_USE_STL
#endif
*/
#include <ticpp.h>


class asFileXml : public asFile
{
public:
    /** Default constructor */
    asFileXml(const wxString &FileName, const ListFileMode &FileMode);

    /** Default destructor */
    virtual ~asFileXml();

    /** Open file */
    virtual bool Open();

    /** Close file */
    virtual bool Close();

    /** Save file */
    bool Save();






    bool GoToFirstNodeWithPath(const wxString node, const int &showWarnings = asSHOW_WARNINGS);

    bool GoToLastNodeWithPath(const wxString node, const int &showWarnings = asSHOW_WARNINGS);

    bool GoToNextSameNode();

    bool GoToNextSameNodeWithAttributeValue(const wxString &attributeName, const wxString &attributeValue);

    bool GoANodeBack();

    bool GoToChildNodeWithAttributeValue(const wxString &AttributeName, const wxString &AttributeValue, const int &showWarnings = asSHOW_WARNINGS);

    wxString GetCurrenNodePath();


    void ClearCurrenNodePath();


    int GetFileRow()
    {
        return m_ElementPointer->Row ();
    }

    int GetFileBaseNodeRow()
    {
        return m_BaseNodePointer->Row ();
    }





    /** Get the element value for the given structure
     * \param ElementAccess The structure to access to a certain node
     * \param ValDefault The default value
     * \return The element value
     */
    template< class T >
    T GetFirstElementValue(const wxString &ElementAccess, const T &ValDefault);

    /** Get the element value for the given structure. Interface for strings
     * \param ElementAccess The structure to access to a certain node
     * \param ValDefault The default value
     * \return The element value
     */
    wxString GetFirstElementValueText(const wxString &ElementAccess, const wxString &ValDefault = wxEmptyString);

    /** Get the element value for the given structure. Interface for ints
     * \param ElementAccess The structure to access to a certain node
     * \param ValDefault The default value
     * \return The element value
     */
    int GetFirstElementValueInt(const wxString &ElementAccess, const int &ValDefault = NaNInt);

    /** Get the element value for the given structure. Interface for floats
     * \param ElementAccess The structure to access to a certain node
     * \param ValDefault The default value
     * \return The element value
     */
    float GetFirstElementValueFloat(const wxString &ElementAccess, const float &ValDefault = NaNFloat);

    /** Get the element value for the given structure. Interface for doubles
     * \param ElementAccess The structure to access to a certain node
     * \param ValDefault The default value
     * \return The element value
     */
    double GetFirstElementValueDouble(const wxString &ElementAccess, const double &ValDefault = NaNDouble);

    /** Get the element value for the given structure. Interface for bools
     * \param ElementAccess The structure to access to a certain node
     * \param ValDefault The default value
     * \return The element value
     */
    bool GetFirstElementValueBool(const wxString &ElementAccess, const bool &ValDefault = false);

    /** Get the next element
     * \param ElementAccess The structure to access to a certain node
     * \return Sibling found or not
     */
    bool GetNextElement(const wxString &ElementAccess);

    /** Get the current element value
     * \param ValDefault The default value
     * \return The element value
     */
    template< class T >
    T GetThisElementValue(const T &ValDefault);

    /** Get the current element value. Interface for strings
     * \param ValDefault The default value
     * \return The element value
     */
    wxString GetThisElementValueText(const wxString &ValDefault = wxEmptyString);

    /** Get the current element value. Interface for ints
     * \param ValDefault The default value
     * \return The element value
     */
    int GetThisElementValueInt(const int &ValDefault = NaNInt);

    /** Get the current element value. Interface for floats
     * \param ValDefault The default value
     * \return The element value
     */
    float GetThisElementValueFloat(const float &ValDefault = NaNFloat);

    /** Get the current element value. Interface for doubles
     * \param ValDefault The default value
     * \return The element value
     */
    double GetThisElementValueDouble(const double &ValDefault = NaNDouble);

    /** Get the current element value. Interface for bools
     * \param ValDefault The default value
     * \return The element value
     */
    bool GetThisElementValueBool(const bool &ValDefault = false);

    /** Get the attribute value for the given structure
     * \param ElementAccess The structure to access to a certain node
     * \param AttributeName The attribute name
     * \param ValDefault The default value
     * \return The element value
     */
    template< class T >
    T GetFirstElementAttributeValue(const wxString &ElementAccess, const wxString &AttributeName, const T &ValDefault);

    /** Get the attribute value for the given structure. Interface for strings
     * \param ElementAccess The structure to access to a certain node
     * \param AttributeName The attribute name
     * \param ValDefault The default value
     * \return The element value
     */
    wxString GetFirstElementAttributeValueText(const wxString &ElementAccess, const wxString &AttributeName, const wxString &ValDefault = wxEmptyString);

    /** Get the attribute value for the given structure. Interface for ints
     * \param ElementAccess The structure to access to a certain node
     * \param AttributeName The attribute name
     * \param ValDefault The default value
     * \return The element value
     */
    int GetFirstElementAttributeValueInt(const wxString &ElementAccess, const wxString &AttributeName, const int &ValDefault = NaNInt);

    /** Get the attribute value for the given structure. Interface for floats
     * \param ElementAccess The structure to access to a certain node
     * \param AttributeName The attribute name
     * \param ValDefault The default value
     * \return The element value
     */
    float GetFirstElementAttributeValueFloat(const wxString &ElementAccess, const wxString &AttributeName, const float &ValDefault = NaNFloat);

    /** Get the attribute value for the given structure. Interface for doubles
     * \param ElementAccess The structure to access to a certain node
     * \param AttributeName The attribute name
     * \param ValDefault The default value
     * \return The element value
     */
    double GetFirstElementAttributeValueDouble(const wxString &ElementAccess, const wxString &AttributeName, const double &ValDefault = NaNDouble);

    /** Get the attribute value for the given structure. Interface for bools
     * \param ElementAccess The structure to access to a certain node
     * \param AttributeName The attribute name
     * \param ValDefault The default value
     * \return The element value
     */
    bool GetFirstElementAttributeValueBool(const wxString &ElementAccess, const wxString &AttributeName, const bool &ValDefault = false);

    /** Get the attribute value for the current element
     * \param AttributeName The attribute name
     * \param ValDefault The default value
     * \return The element value
     */
    template< class T >
    T GetThisElementAttributeValue(const wxString &AttributeName, const T &ValDefault);

    /** Get the attribute value for the current element. Interface for strings
     * \param AttributeName The attribute name
     * \param ValDefault The default value
     * \return The element value
     */
    wxString GetThisElementAttributeValueText(const wxString &AttributeName, const wxString &ValDefault = wxEmptyString);

    /** Get the attribute value for the current element. Interface for ints
     * \param AttributeName The attribute name
     * \param ValDefault The default value
     * \return The element value
     */
    int GetThisElementAttributeValueInt(const wxString &AttributeName, const int &ValDefault = NaNInt);

    /** Get the attribute value for the current element. Interface for floats
     * \param AttributeName The attribute name
     * \param ValDefault The default value
     * \return The element value
     */
    float GetThisElementAttributeValueFloat(const wxString &AttributeName, const float &ValDefault = NaNFloat);

    /** Get the attribute value for the current element. Interface for doubles
     * \param AttributeName The attribute name
     * \param ValDefault The default value
     * \return The element value
     */
    double GetThisElementAttributeValueDouble(const wxString &AttributeName, const double &ValDefault = NaNDouble);

    /** Get the attribute value for the current element. Interface for bools
     * \param AttributeName The attribute name
     * \param ValDefault The default value
     * \return The element value
     */
    bool GetThisElementAttributeValueBool(const wxString &AttributeName, const bool &ValDefault = false);

    /** Inserts an element at the end inside another node
     * \param ParentNodeName The parent element
     * \param NewNodeName The new element
     */
    bool InsertElement(const wxString &ParentNodeName, const wxString &NewNodeName );





    bool InsertElementAndValue(const wxString &ParentNodeName, const wxString &NewNodeName, const wxString &Value );

    bool InsertElementAndAttribute(const wxString &ParentNodeName, const wxString &NewNodeName, const wxString &AttributeName, const wxString &AttributeValue );

    bool SetElementAttribute(const wxString &ParentNodeName, const wxString &AttributeName, const wxString &AttributeValue );

    bool SetElementValue(const wxString &ParentNodeName, const wxString &Value );


protected:
private:
    ticpp::Document* m_Document;
    ticpp::Element* m_ElementPointer;
    ticpp::Element* m_BaseNodePointer;
    wxString m_ElementName;
    wxString m_BaseNodeName;

    /** Check for the presence of a certain markup in the xml file at the current node level
    * \param pElem The node element
    * \param MarkupName The markup name to find
    * \return Found or not
    */
    bool MarkupExists(const ticpp::Element* pElem, const wxString &MarkupName);

    /** Find a certain markup in the xml file at the current node level
     * \param pElem The node element
     * \param MarkupName The markup name to find
     * \return The desired child node
     */
    ticpp::Element* FindMarkup(const ticpp::Element* pElem, const wxString &MarkupName);

    /** Find a certain markup in the xml file at the current node level according to an attribute value
     * \param pElem The node element
     * \param AttributeName The attribute name to consider
     * \param AttributeValue The attribute value to find
     * \return The desired child node
     */
    ticpp::Element* GetChildFromAttributeValue(const ticpp::Element* pElem, const wxString &AttributeName, const wxString &AttributeValue);

    /** Find a certain markup in the xml file according to the given structure
     * \param ElementAccessConst The structure to access to a certain node
     */
    bool SetPointerFirstElement(const wxString &ElementAccessConst);

    /** Find a certain markup in the xml file according to the given structure
     * \param ElementAccessConst The structure to access to a certain node
     */
    bool SetPointerLastElement(const wxString &ElementAccessConst);

    /** Find the next sibling element in the xml file according to the given structure
     * \param throwIfNoSiblings If true, will throw an exception if there are no sibling element.
     */
    bool SetPointerNextSameSibling(const bool &throwIfNoSiblings = false);



    bool UpdateElementName();

};

#endif
