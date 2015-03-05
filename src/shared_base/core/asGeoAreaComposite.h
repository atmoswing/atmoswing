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


    /** Access m_cornerUL
     * \return The current value of m_cornerUL
     */
    Coo GetCornerUL()
    {
        return m_cornerUL;
    }

    /** Access m_cornerUR
     * \return The current value of m_cornerUR
     */
    Coo GetCornerUR()
    {
        return m_cornerUR;
    }

    /** Access m_cornerLL
     * \return The current value of m_cornerLL
     */
    Coo GetCornerLL()
    {
        return m_cornerLL;
    }

    /** Access m_cornerLR
     * \return The current value of m_cornerLR
     */
    Coo GetCornerLR()
    {
        return m_cornerLR;
    }

    /** Access m_level
     * \return The current value of m_level
     */
    double GetLevel()
    {
        return m_level;
    }

    /** Set m_level
     * \param val New value to set
     */
    void SetLevel(double val)
    {
        m_level = val;
    }

    /** Access m_absoluteXmin
     * \return The current value of m_absoluteXmin
     */
    double GetAbsoluteXmin()
    {
        return m_absoluteXmin;
    }

    /** Access m_absolutXmax
     * \return The current value of m_absoluteXmax
     */
    double GetAbsoluteXmax()
    {
        return m_absoluteXmax;
    }

    /** Access m_absoluteYmin
     * \return The current value of m_absoluteYmin
     */
    double GetAbsoluteYmin()
    {
        return m_absoluteYmin;
    }

    /** Access m_absoluteYmax
     * \return The current value of m_absoluteYmax
     */
    double GetAbsoluteYmax()
    {
        return m_absoluteYmax;
    }

    /** Gives the area absolute X width
     * \return The area absolute X width
     */
    double GetAbsoluteXwidth()
    {
        return abs(m_absoluteXmax-m_absoluteXmin);
    }

    /** Gives the area absolute Y width
     * \return The area absolute Y width
     */
    double GetAbsoluteYwidth()
    {
        return abs(m_absoluteYmax-m_absoluteYmin);
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
        return m_nbComposites;
    }

    /** Gives the composite areas
     * \return The composite areas
     */
    vector <asGeoArea> GetComposites()
    {
        return m_composites;
    }

    /** Gives a specific composite area
     * \return The composite area desired
     */
    asGeoArea GetComposite(int Id)
    {
        if(Id>=m_nbComposites) asThrowException(_("The composite area doesn't exist."));
        return m_composites[Id];
    }

    /** Tells if the area is a straight rectangle or not
     * \return True if the area is a straight rectangle
     */
    bool IsRectangle();

protected:
    /** Creates the composites */
    void CreateComposites();

private:
    double m_absoluteXmin;
    double m_absoluteXmax;
    double m_absoluteYmin;
    double m_absoluteYmax;
    Coo m_cornerUL; //!< Member variable "m_cornerUL"
    Coo m_cornerUR; //!< Member variable "m_cornerUR"
    Coo m_cornerLL; //!< Member variable "m_cornerDL"
    Coo m_cornerLR; //!< Member variable "m_cornerDR"
    float m_level; //!< Member variable "m_level" hPa
    float m_height; //!< Member variable "m_height" m
    int m_nbComposites; //!< Member variable "m_nbComposites"
    vector <asGeoArea> m_composites; //!< Member variable "m_composites"
    int m_flatAllowed; //!< Member variable "m_flatAllowed"

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
