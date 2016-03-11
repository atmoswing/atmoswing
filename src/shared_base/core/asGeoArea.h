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
 
#ifndef ASGEOAREA_H
#define ASGEOAREA_H

#include <asIncludes.h>
#include <asGeo.h>

class asGeoArea: public asGeo
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
    asGeoArea(const Coo &CornerUL, const Coo &CornerUR, const Coo &CornerLL, const Coo &CornerLR, float Level = asNONE, float Height = asNONE, int flatAllowed = asFLAT_FORBIDDEN);

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
    asGeoArea(double Xmin, double Xwidth, double Ymin, double Ywidth, float Level = asNONE, float Height = asNONE, int flatAllowed = asFLAT_FORBIDDEN);


    asGeoArea(float Level = asNONE, float Height = asNONE);


    /** Default destructor */
    virtual ~asGeoArea();


    void Generate(double Xmin, double Xwidth, double Ymin, double Ywidth, int flatAllowed = asFLAT_FORBIDDEN);

    /** Access m_cornerUL
     * \return The current value of m_cornerUL
     */
    Coo GetCornerUL()
    {
        return m_cornerUL;
    }

    /** Set m_cornerUL
     * \param val New value to set
     */
    void SetCornerUL(const Coo &val)
    {
        m_cornerUL = val;
        Init();
    }

    /** Access m_cornerUR
     * \return The current value of m_cornerUR
     */
    Coo GetCornerUR()
    {
        return m_cornerUR;
    }

    /** Set m_cornerUR
     * \param val New value to set
     */
    void SetCornerUR(const Coo &val)
    {
        m_cornerUR = val;
        Init();
    }

    /** Access m_cornerLL
     * \return The current value of m_cornerLL
     */
    Coo GetCornerLL()
    {
        return m_cornerLL;
    }

    /** Set m_cornerLL
     * \param val New value to set
     */
    void SetCornerLL(const Coo &val)
    {
        m_cornerLL = val;
        Init();
    }

    /** Access m_cornerLR
     * \return The current value of m_cornerLR
     */
    Coo GetCornerLR()
    {
        return m_cornerLR;
    }

    /** Set m_cornerLR
     * \param val New value to set
     */
    void SetCornerLR(const Coo &val)
    {
        m_cornerLR = val;
        Init();
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

    /** Gives the area X min coordinate
     * \return The value of the minimum on the X axis
     */
    double GetXmin();

    /** Gives the area X max coordinate
     * \return The value of the maximum on the X axis
     */
    double GetXmax();

    /** Gives the area X size
     * \return The value of the X axis size
     */
    double GetXwidth();

    /** Gives the area Y min coordinate
     * \return The value of the minimum on the Y axis
     */
    double GetYmin();

    /** Gives the area Y max coordinate
     * \return The value of the maximum on the Y axis
     */
    double GetYmax();

    /** Gives the area Y size
     * \return The value of the Y axis size
     */
    double GetYwidth();

    /** Gives the area center coordinates
     * \return The coordinates of the center
     */
    Coo GetCenter();

    /** Tells if the area is a straight rectangle or not
     * \return True if the area is a straight rectangle
     */
    bool IsRectangle();

protected:
    Coo m_cornerUL; //!< Member variable "m_cornerUL"
    Coo m_cornerUR; //!< Member variable "m_cornerUR"
    Coo m_cornerLL; //!< Member variable "m_cornerDL"
    Coo m_cornerLR; //!< Member variable "m_cornerDR"
    float m_level; //!< Member variable "m_level" hPa
    float m_height; //!< Member variable "m_height" m
    int m_flatAllowed; //!< Member variable "m_flatAllowed"

private:

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
