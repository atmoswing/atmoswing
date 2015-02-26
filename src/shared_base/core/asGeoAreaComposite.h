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
 
#ifndef ASGEOAREACOMPOSITE_H
#define ASGEOAREACOMPOSITE_H

#include <asIncludes.h>
#include <asGeoArea.h>

class asGeoAreaComposite: public asGeo
{
public:

    /** Default constructor
     * \param coosys The coordinate system
     * \param CornerUL The coordinates of the upper left corner
     * \param CornerUR The coordinates of the upper right corner
     * \param CornerLL The coordinates of the lower left corner
     * \param CornerLR The coordinates of the lower right corner
     * \param Level The height in hPa
     * \param Height The height in m
     * \param flatAllowed Allows the area to have a dimension that is null
     */
    asGeoAreaComposite(const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL, const Coo &CornerLR, float Level = asNONE, float Height = asNONE, int flatAllowed = asFLAT_FORBIDDEN);

    /** Alternative constructor
     * \param coosys The coordinate system
     * \param Xmin The left border
     * \param Xwidth The size on X axis
     * \param Ymin The left border
     * \param Ywidth The size on Y axis
     * \param Level The height in hPa
     * \param Height The height in m
     * \param flatAllowed Allows the area to have a dimension that is null
     */
    asGeoAreaComposite(double Xmin, double Xwidth, double Ymin, double Ywidth, float Level = asNONE, float Height = asNONE, int flatAllowed = asFLAT_FORBIDDEN);


    asGeoAreaComposite(float Level = asNONE, float Height = asNONE);


    /** Default destructor */
    virtual ~asGeoAreaComposite();


    void Generate(double Xmin, double Xwidth, double Ymin, double Ywidth, int flatAllowed = asFLAT_FORBIDDEN);


    /** Access m_CornerUL
     * \return The current value of m_CornerUL
     */
    Coo GetCornerUL()
    {
        return m_CornerUL;
    }

    /** Access m_CornerUR
     * \return The current value of m_CornerUR
     */
    Coo GetCornerUR()
    {
        return m_CornerUR;
    }

    /** Access m_CornerLL
     * \return The current value of m_CornerLL
     */
    Coo GetCornerLL()
    {
        return m_CornerLL;
    }

    /** Access m_CornerLR
     * \return The current value of m_CornerLR
     */
    Coo GetCornerLR()
    {
        return m_CornerLR;
    }

    /** Access m_Level
     * \return The current value of m_Level
     */
    double GetLevel()
    {
        return m_Level;
    }

    /** Set m_Level
     * \param val New value to set
     */
    void SetLevel(double val)
    {
        m_Level = val;
    }

    /** Access m_AbsoluteXmin
     * \return The current value of m_AbsoluteXmin
     */
    double GetAbsoluteXmin()
    {
        return m_AbsoluteXmin;
    }

    /** Access m_AbsolutXmax
     * \return The current value of m_AbsoluteXmax
     */
    double GetAbsoluteXmax()
    {
        return m_AbsoluteXmax;
    }

    /** Access m_AbsoluteYmin
     * \return The current value of m_AbsoluteYmin
     */
    double GetAbsoluteYmin()
    {
        return m_AbsoluteYmin;
    }

    /** Access m_AbsoluteYmax
     * \return The current value of m_AbsoluteYmax
     */
    double GetAbsoluteYmax()
    {
        return m_AbsoluteYmax;
    }

    /** Gives the area absolute X width
     * \return The area absolute X width
     */
    double GetAbsoluteXwidth()
    {
        return abs(m_AbsoluteXmax-m_AbsoluteXmin);
    }

    /** Gives the area absolute Y width
     * \return The area absolute Y width
     */
    double GetAbsoluteYwidth()
    {
        return abs(m_AbsoluteYmax-m_AbsoluteYmin);
    }

    /** Gives the area X min coordinate
     * \return The value of the minimum on the X axis
     */
    double GetXmin();

    /** Gives the area X max coordinate
     * \return The value of the maximum on the X axis
     */
    double GetXmax();

    /** Gives the area Y min coordinate
     * \return The value of the minimum on the Y axis
     */
    double GetYmin();

    /** Gives the area Y max coordinate
     * \return The value of the maximum on the Y axis
     */
    double GetYmax();

    /** Gives the area center coordinates
     * \return The coordinates of the center
     */
    Coo GetCenter();

    /** Gives the number of composite areas
     * \return The number of composite areas
     */
    int GetNbComposites()
    {
        return m_NbComposites;
    }

    /** Gives the composite areas
     * \return The composite areas
     */
    vector <asGeoArea> GetComposites()
    {
        return m_Composites;
    }

    /** Gives a specific composite area
     * \return The composite area desired
     */
    asGeoArea GetComposite(int Id)
    {
        if(Id>=m_NbComposites) asThrowException(_("The composite area doesn't exist."));
        return m_Composites[Id];
    }

    /** Tells if the area is a straight rectangle or not
     * \return True if the area is a straight rectangle
     */
    bool IsRectangle();

protected:
    /** Creates the composites */
    void CreateComposites();

private:
    double m_AbsoluteXmin;
    double m_AbsoluteXmax;
    double m_AbsoluteYmin;
    double m_AbsoluteYmax;
    Coo m_CornerUL; //!< Member variable "m_CornerUL"
    Coo m_CornerUR; //!< Member variable "m_CornerUR"
    Coo m_CornerLL; //!< Member variable "m_CornerDL"
    Coo m_CornerLR; //!< Member variable "m_CornerDR"
    float m_Level; //!< Member variable "m_Level" hPa
    float m_Height; //!< Member variable "m_Height" m
    int m_NbComposites; //!< Member variable "m_NbComposites"
    vector <asGeoArea> m_Composites; //!< Member variable "m_Composites"
    int m_FlatAllowed; //!< Member variable "m_FlatAllowed"

    /** Process to initialization and checks */
    void Init();

    /** Check every point
     * \return True on success
     */
    bool DoCheckPoints();

    /** Check the consistency of the points
     * \return True on success
     */
    bool CheckConsistency();
};

#endif
