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

    /** Access m_CornerUL
     * \return The current value of m_CornerUL
     */
    Coo GetCornerUL()
    {
        return m_CornerUL;
    }

    /** Set m_CornerUL
     * \param val New value to set
     */
    void SetCornerUL(const Coo &val)
    {
        m_CornerUL = val;
        Init();
    }

    /** Access m_CornerUR
     * \return The current value of m_CornerUR
     */
    Coo GetCornerUR()
    {
        return m_CornerUR;
    }

    /** Set m_CornerUR
     * \param val New value to set
     */
    void SetCornerUR(const Coo &val)
    {
        m_CornerUR = val;
        Init();
    }

    /** Access m_CornerLL
     * \return The current value of m_CornerLL
     */
    Coo GetCornerLL()
    {
        return m_CornerLL;
    }

    /** Set m_CornerLL
     * \param val New value to set
     */
    void SetCornerLL(const Coo &val)
    {
        m_CornerLL = val;
        Init();
    }

    /** Access m_CornerLR
     * \return The current value of m_CornerLR
     */
    Coo GetCornerLR()
    {
        return m_CornerLR;
    }

    /** Set m_CornerLR
     * \param val New value to set
     */
    void SetCornerLR(const Coo &val)
    {
        m_CornerLR = val;
        Init();
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
    Coo m_CornerUL; //!< Member variable "m_CornerUL"
    Coo m_CornerUR; //!< Member variable "m_CornerUR"
    Coo m_CornerLL; //!< Member variable "m_CornerDL"
    Coo m_CornerLR; //!< Member variable "m_CornerDR"
    float m_Level; //!< Member variable "m_Level" hPa
    float m_Height; //!< Member variable "m_Height" m
    int m_FlatAllowed; //!< Member variable "m_FlatAllowed"

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
