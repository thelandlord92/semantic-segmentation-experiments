#include <E57SimpleReader.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// Each scanner pose should have at most six camera/cube images.
// If an E57 contains more than six Image2D blocks, the extra ones are skipped.
static constexpr int MAX_IMAGES_PER_POSE = 6;

// Escape characters that would otherwise break JSON string values.
std::string jsonEscape(const std::string& s)
{
    std::ostringstream out;

    for (char c : s)
    {
        switch (c)
        {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default: out << c; break;
        }
    }

    return out.str();
}

// Convenience wrapper for writing a quoted JSON string.
std::string q(const std::string& s)
{
    return "\"" + jsonEscape(s) + "\"";
}

// Convert strings to lowercase so file extension checks are case insensitive.
std::string lowerString(std::string s)
{
    std::transform(
        s.begin(),
        s.end(),
        s.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    return s;
}

// Return true only for regular files ending in .e57.
bool isE57File(const fs::path& p)
{
    return fs::is_regular_file(p) && lowerString(p.extension().string()) == ".e57";
}

// Convert libE57Format's projection enum to a readable JSON string.
std::string projectionToString(e57::Image2DProjection p)
{
    switch (p)
    {
    case e57::ProjectionVisual: return "visual";
    case e57::ProjectionPinhole: return "pinhole";
    case e57::ProjectionSpherical: return "spherical";
    case e57::ProjectionCylindrical: return "cylindrical";
    default: return "none";
    }
}

// Convert libE57Format's image type enum to a readable JSON string.
std::string imageTypeToString(e57::Image2DType t)
{
    switch (t)
    {
    case e57::ImageJPEG: return "jpeg";
    case e57::ImagePNG: return "png";
    case e57::ImageMaskPNG: return "png_mask";
    default: return "none";
    }
}

// Use the image type to choose the correct extension for the extracted image file.
std::string imageExtension(e57::Image2DType t)
{
    switch (t)
    {
    case e57::ImageJPEG: return ".jpg";
    case e57::ImagePNG: return ".png";
    case e57::ImageMaskPNG: return ".png";
    default: return ".bin";
    }
}

// Convert an E57 quaternion into a 3x3 rotation matrix.
// E57 stores rotations as quaternions in w, x, y, z order.
std::array<std::array<double, 3>, 3> quaternionToMatrix(const e57::Quaternion& qin)
{
    double w = qin.w;
    double x = qin.x;
    double y = qin.y;
    double z = qin.z;

    // Normalize defensively, in case the stored quaternion has small numerical drift.
    const double n = std::sqrt(w * w + x * x + y * y + z * z);
    if (n > 0.0)
    {
        w /= n;
        x /= n;
        y /= n;
        z /= n;
    }

    // Standard quaternion-to-rotation-matrix conversion.
    return {{
        {{1.0 - 2.0 * y * y - 2.0 * z * z, 2.0 * x * y - 2.0 * z * w,       2.0 * x * z + 2.0 * y * w}},
        {{2.0 * x * y + 2.0 * z * w,       1.0 - 2.0 * x * x - 2.0 * z * z, 2.0 * y * z - 2.0 * x * w}},
        {{2.0 * x * z - 2.0 * y * w,       2.0 * y * z + 2.0 * x * w,       1.0 - 2.0 * x * x - 2.0 * y * y}}
    }};
}

// Convert an E57 rigid-body transform into a 4x4 homogeneous matrix.
// This is useful in Python because it can be used directly with NumPy.
std::array<std::array<double, 4>, 4> transformMatrix(const e57::RigidBodyTransform& t)
{
    auto R = quaternionToMatrix(t.rotation);

    return {{
        {{R[0][0], R[0][1], R[0][2], t.translation.x}},
        {{R[1][0], R[1][1], R[1][2], t.translation.y}},
        {{R[2][0], R[2][1], R[2][2], t.translation.z}},
        {{0.0,     0.0,     0.0,     1.0}}
    }};
}

// Transform a 3D point from the local scan coordinate system to the world/project system.
// This is mainly used here to compute a world-space centre for each scan bounding box.
std::array<double, 3> transformPoint(
    const e57::RigidBodyTransform& t,
    const std::array<double, 3>& p)
{
    auto R = quaternionToMatrix(t.rotation);

    return {{
        R[0][0] * p[0] + R[0][1] * p[1] + R[0][2] * p[2] + t.translation.x,
        R[1][0] * p[0] + R[1][1] * p[1] + R[1][2] * p[2] + t.translation.y,
        R[2][0] * p[0] + R[2][1] * p[1] + R[2][2] * p[2] + t.translation.z
    }};
}

// Write a 3-element numeric array to JSON.
void writeArray3(std::ostream& os, const std::array<double, 3>& a)
{
    os << "[" << a[0] << ", " << a[1] << ", " << a[2] << "]";
}

// Write a 3x3 numeric matrix to JSON.
void writeMatrix3(std::ostream& os, const std::array<std::array<double, 3>, 3>& m)
{
    os << "[";
    for (int i = 0; i < 3; ++i)
    {
        if (i > 0) os << ", ";
        os << "[" << m[i][0] << ", " << m[i][1] << ", " << m[i][2] << "]";
    }
    os << "]";
}

// Write a 4x4 numeric matrix to JSON.
void writeMatrix4(std::ostream& os, const std::array<std::array<double, 4>, 4>& m)
{
    os << "[";
    for (int i = 0; i < 4; ++i)
    {
        if (i > 0) os << ", ";
        os << "[" << m[i][0] << ", " << m[i][1] << ", " << m[i][2] << ", " << m[i][3] << "]";
    }
    os << "]";
}

// Write a full pose object to JSON.
// The same function is used for scan poses and camera poses.
void writeTransformJson(std::ostream& os, const e57::RigidBodyTransform& pose, int indent)
{
    const std::string sp(indent, ' ');
    auto R = quaternionToMatrix(pose.rotation);
    auto T = transformMatrix(pose);

    os << sp << "{\n";
    os << sp << "  \"translation_m\": {\n";
    os << sp << "    \"x\": " << pose.translation.x << ",\n";
    os << sp << "    \"y\": " << pose.translation.y << ",\n";
    os << sp << "    \"z\": " << pose.translation.z << "\n";
    os << sp << "  },\n";

    // Store the raw E57 quaternion so no information is lost.
    os << sp << "  \"rotation_quaternion_wxyz\": {\n";
    os << sp << "    \"w\": " << pose.rotation.w << ",\n";
    os << sp << "    \"x\": " << pose.rotation.x << ",\n";
    os << sp << "    \"y\": " << pose.rotation.y << ",\n";
    os << sp << "    \"z\": " << pose.rotation.z << "\n";
    os << sp << "  },\n";

    // Store derived matrix forms for easier use in Python projection code.
    os << sp << "  \"rotation_matrix_3x3\": ";
    writeMatrix3(os, R);
    os << ",\n";
    os << sp << "  \"transform_matrix_4x4\": ";
    writeMatrix4(os, T);
    os << "\n";
    os << sp << "}";
}

// Write scan bounding-box information to JSON.
// The local bounds come from the E57 file. The world centre is computed using the scan pose.
void writeBoundsJson(
    std::ostream& os,
    const e57::CartesianBounds& b,
    const e57::RigidBodyTransform& pose,
    int indent)
{
    const std::string sp(indent, ' ');

    const double sx = b.xMaximum - b.xMinimum;
    const double sy = b.yMaximum - b.yMinimum;
    const double sz = b.zMaximum - b.zMinimum;

    std::array<double, 3> centerLocal = {{
        0.5 * (b.xMinimum + b.xMaximum),
        0.5 * (b.yMinimum + b.yMaximum),
        0.5 * (b.zMinimum + b.zMaximum)
    }};

    auto centerWorld = transformPoint(pose, centerLocal);

    os << sp << "{\n";
    os << sp << "  \"bounds_local_m\": {\n";
    os << sp << "    \"x_minimum\": " << b.xMinimum << ",\n";
    os << sp << "    \"x_maximum\": " << b.xMaximum << ",\n";
    os << sp << "    \"y_minimum\": " << b.yMinimum << ",\n";
    os << sp << "    \"y_maximum\": " << b.yMaximum << ",\n";
    os << sp << "    \"z_minimum\": " << b.zMinimum << ",\n";
    os << sp << "    \"z_maximum\": " << b.zMaximum << "\n";
    os << sp << "  },\n";

    // Box dimensions are helpful for quick sanity checks in Python.
    os << sp << "  \"box_size_m\": ";
    writeArray3(os, {{sx, sy, sz}});
    os << ",\n";

    // Local centre is useful if you work in the scan's native coordinate frame.
    os << sp << "  \"box_center_local_m\": ";
    writeArray3(os, centerLocal);
    os << ",\n";

    // World centre is useful if the scan has already been registered globally.
    os << sp << "  \"box_center_world_m\": ";
    writeArray3(os, centerWorld);
    os << "\n";
    os << sp << "}";
}

// Write one image's metadata to JSON.
// This includes the saved filename, image pose, image dimensions, and all camera-model data exposed by libE57Format.
void writeImageJson(
    std::ostream& os,
    const e57::Image2D& img,
    const std::string& savedFileName,
    e57::Image2DProjection projection,
    e57::Image2DType imageType,
    int64_t imageWidth,
    int64_t imageHeight,
    int64_t imageSize,
    int indent)
{
    const std::string sp(indent, ' ');

    os << sp << "{\n";
    os << sp << "  \"file_name\": " << q(savedFileName) << ",\n";
    os << sp << "  \"name\": " << q(img.name) << ",\n";
    os << sp << "  \"guid\": " << q(img.guid) << ",\n";
    os << sp << "  \"associated_data3d_guid\": " << q(img.associatedData3DGuid) << ",\n";
    os << sp << "  \"projection\": " << q(projectionToString(projection)) << ",\n";
    os << sp << "  \"image_type\": " << q(imageTypeToString(imageType)) << ",\n";
    os << sp << "  \"image_width\": " << imageWidth << ",\n";
    os << sp << "  \"image_height\": " << imageHeight << ",\n";
    os << sp << "  \"image_size_bytes\": " << imageSize << ",\n";

    // The camera centre is the translation component of the image pose.
    os << sp << "  \"camera_center_world_m\": ["
       << img.pose.translation.x << ", "
       << img.pose.translation.y << ", "
       << img.pose.translation.z << "],\n";

    // Full camera pose needed for projecting 3D points into this image.
    os << sp << "  \"camera_pose\": ";
    writeTransformJson(os, img.pose, indent + 2);
    os << ",\n";

    // Pinhole fields are the important ones for cube-map/perspective images.
    // fx and fy are derived in pixels so Python does not need to repeat this step.
    os << sp << "  \"pinhole\": {\n";
    os << sp << "    \"image_width\": " << img.pinholeRepresentation.imageWidth << ",\n";
    os << sp << "    \"image_height\": " << img.pinholeRepresentation.imageHeight << ",\n";
    os << sp << "    \"focal_length_m\": " << img.pinholeRepresentation.focalLength << ",\n";
    os << sp << "    \"pixel_width_m\": " << img.pinholeRepresentation.pixelWidth << ",\n";
    os << sp << "    \"pixel_height_m\": " << img.pinholeRepresentation.pixelHeight << ",\n";
    os << sp << "    \"principal_point_x_px\": " << img.pinholeRepresentation.principalPointX << ",\n";
    os << sp << "    \"principal_point_y_px\": " << img.pinholeRepresentation.principalPointY << ",\n";
    os << sp << "    \"fx_px\": "
       << (img.pinholeRepresentation.pixelWidth > 0.0
           ? img.pinholeRepresentation.focalLength / img.pinholeRepresentation.pixelWidth
           : 0.0)
       << ",\n";
    os << sp << "    \"fy_px\": "
       << (img.pinholeRepresentation.pixelHeight > 0.0
           ? img.pinholeRepresentation.focalLength / img.pinholeRepresentation.pixelHeight
           : 0.0)
       << "\n";
    os << sp << "  },\n";

    // Spherical fields are included in case a file contains true panoramic Image2D entries.
    os << sp << "  \"spherical\": {\n";
    os << sp << "    \"image_width\": " << img.sphericalRepresentation.imageWidth << ",\n";
    os << sp << "    \"image_height\": " << img.sphericalRepresentation.imageHeight << ",\n";
    os << sp << "    \"pixel_width_rad\": " << img.sphericalRepresentation.pixelWidth << ",\n";
    os << sp << "    \"pixel_height_rad\": " << img.sphericalRepresentation.pixelHeight << "\n";
    os << sp << "  },\n";

    // Cylindrical fields are included for completeness, although Leica cube-map images are likely pinhole.
    os << sp << "  \"cylindrical\": {\n";
    os << sp << "    \"image_width\": " << img.cylindricalRepresentation.imageWidth << ",\n";
    os << sp << "    \"image_height\": " << img.cylindricalRepresentation.imageHeight << ",\n";
    os << sp << "    \"pixel_width_rad\": " << img.cylindricalRepresentation.pixelWidth << ",\n";
    os << sp << "    \"pixel_height_m\": " << img.cylindricalRepresentation.pixelHeight << ",\n";
    os << sp << "    \"radius_m\": " << img.cylindricalRepresentation.radius << ",\n";
    os << sp << "    \"principal_point_y_px\": " << img.cylindricalRepresentation.principalPointY << "\n";
    os << sp << "  }\n";

    os << sp << "}";
}

// Stores the metadata for each image that was successfully extracted.
// These records are written to the JSON after the image files are saved.
struct SavedImageRecord
{
    e57::Image2D header;
    std::string fileName;
    e57::Image2DProjection projection = e57::ProjectionNone;
    e57::Image2DType imageType = e57::ImageNone;
    int64_t width = 0;
    int64_t height = 0;
    int64_t size = 0;
};

int main(int argc, char** argv)
{
    // Expected command:
    // e57_folder_pose_exporter "D:\path\to\folder_with_individual_e57_scans"
    if (argc < 2)
    {
        std::cerr << "Usage:\n";
        std::cerr << "  e57_folder_pose_exporter input_folder\n";
        return 1;
    }

    const fs::path inputFolder = argv[1];

    // Output files are written inside the same folder that contains the individual E57 scans.
    const fs::path imagesFolder = inputFolder / "pose_images";
    const fs::path jsonPath = inputFolder / "pose_data.json";

    if (!fs::exists(inputFolder) || !fs::is_directory(inputFolder))
    {
        std::cerr << "Input folder does not exist or is not a directory.\n";
        return 1;
    }

    fs::create_directories(imagesFolder);

    // Collect all E57 files in the input folder.
    // The files are sorted so pose numbering is deterministic.
    std::vector<fs::path> e57Files;
    for (const auto& entry : fs::directory_iterator(inputFolder))
    {
        if (isE57File(entry.path()))
        {
            e57Files.push_back(entry.path());
        }
    }

    std::sort(e57Files.begin(), e57Files.end());

    if (e57Files.empty())
    {
        std::cerr << "No .e57 files found in folder: " << inputFolder << "\n";
        return 1;
    }

    // Create one combined JSON file for the full folder.
    std::ofstream json(jsonPath);
    json << std::fixed << std::setprecision(12);

    json << "{\n";
    json << "  \"source_folder\": " << q(inputFolder.string()) << ",\n";
    json << "  \"pose_images_folder\": " << q(imagesFolder.string()) << ",\n";
    json << "  \"pose_count\": " << e57Files.size() << ",\n";
    json << "  \"poses\": [\n";

    int poseNumber = 0;

    for (const fs::path& e57Path : e57Files)
    {
        ++poseNumber;
        const std::string poseId = "pose" + std::to_string(poseNumber);

        std::cout << "Processing " << poseId << ": " << e57Path.filename().string() << "\n";

        // Open the current E57 file.
        // This assumes each E57 file represents one scan position or scanner pose.
        e57::Reader reader(e57Path.string(), e57::ReaderOptions{});

        const int64_t data3DCount = reader.GetData3DCount();
        const int64_t image2DCount = reader.GetImage2DCount();

        e57::Data3D scan;
        bool hasScan = false;

        // Variables returned by GetData3DSizes.
        // pointsSize is the number of point records in the scan.
        int64_t rowMax = 0;
        int64_t columnMax = 0;
        int64_t pointsSize = 0;
        int64_t groupsSize = 0;
        int64_t countSize = 0;
        bool columnIndex = false;

        // Use the first Data3D block as the pose scan.
        // If the file has more than one Data3D block, this is noted in the warnings section.
        if (data3DCount > 0)
        {
            hasScan = reader.ReadData3D(0, scan);
            if (hasScan)
            {
                reader.GetData3DSizes(
                    0,
                    rowMax,
                    columnMax,
                    pointsSize,
                    groupsSize,
                    countSize,
                    columnIndex);

                scan.pointCount = static_cast<size_t>(pointsSize);
            }
        }

        std::vector<SavedImageRecord> savedImages;
        std::vector<std::string> poseImageNames;
        int savedImageCount = 0;

        // Extract up to six Image2D blocks from this E57 file.
        for (int64_t imageIndex = 0; imageIndex < image2DCount; ++imageIndex)
        {
            if (savedImageCount >= MAX_IMAGES_PER_POSE)
            {
                break;
            }

            e57::Image2D img;
            if (!reader.ReadImage2D(imageIndex, img))
            {
                continue;
            }

            // If the image declares an associated Data3D GUID, keep only images attached to the scan.
            // If the association is empty, the image is not rejected because some exporters omit this link.
            if (hasScan && !img.associatedData3DGuid.empty() && img.associatedData3DGuid != scan.guid)
            {
                continue;
            }

            // Get the actual image blob type and size.
            e57::Image2DProjection projection = e57::ProjectionNone;
            e57::Image2DType imageType = e57::ImageNone;
            e57::Image2DType imageMaskType = e57::ImageNone;
            e57::Image2DType imageVisualType = e57::ImageNone;
            int64_t width = 0;
            int64_t height = 0;
            int64_t imageSize = 0;

            const bool gotSizes = reader.GetImage2DSizes(
                imageIndex,
                projection,
                imageType,
                width,
                height,
                imageSize,
                imageMaskType,
                imageVisualType);

            if (!gotSizes || imageType == e57::ImageNone || imageSize <= 0)
            {
                continue;
            }

            // Read the raw encoded image bytes, usually JPEG or PNG.
            std::vector<uint8_t> buffer(static_cast<size_t>(imageSize));

            const int64_t bytesRead = reader.ReadImage2DData(
                imageIndex,
                projection,
                imageType,
                buffer.data(),
                0,
                imageSize);

            if (bytesRead <= 0)
            {
                continue;
            }

            ++savedImageCount;

            // Name convention requested by the user:
            // pose1_image1.jpg, pose1_image2.jpg, pose30_image3.jpg, etc.
            const std::string imageFileName =
                poseId + "_image" + std::to_string(savedImageCount) + imageExtension(imageType);

            const fs::path imageOutPath = imagesFolder / imageFileName;

            // Save the encoded image bytes exactly as stored in the E57.
            std::ofstream imageOut(imageOutPath, std::ios::binary);
            imageOut.write(
                reinterpret_cast<const char*>(buffer.data()),
                static_cast<std::streamsize>(bytesRead));
            imageOut.close();

            // Keep metadata for the JSON output.
            SavedImageRecord rec;
            rec.header = img;
            rec.fileName = imageFileName;
            rec.projection = projection;
            rec.imageType = imageType;
            rec.width = width;
            rec.height = height;
            rec.size = bytesRead;

            savedImages.push_back(rec);
            poseImageNames.push_back(imageFileName);

            std::cout << "  saved " << imageFileName << "\n";
        }

        reader.Close();

        // Add a comma between pose objects in the JSON array.
        if (poseNumber > 1)
        {
            json << ",\n";
        }

        json << "    {\n";
        json << "      \"pose_id\": " << q(poseId) << ",\n";
        json << "      \"source_e57_file\": " << q(e57Path.filename().string()) << ",\n";
        json << "      \"source_e57_path\": " << q(e57Path.string()) << ",\n";
        json << "      \"source_file_size_bytes\": " << static_cast<uint64_t>(fs::file_size(e57Path)) << ",\n";
        json << "      \"inspection\": {\n";
        json << "        \"data3d_count\": " << data3DCount << ",\n";
        json << "        \"image2d_count\": " << image2DCount << ",\n";
        json << "        \"used_data3d_index\": " << (hasScan ? 0 : -1) << ",\n";
        json << "        \"saved_image_count\": " << savedImages.size() << ",\n";
        json << "        \"max_images_per_pose\": " << MAX_IMAGES_PER_POSE << "\n";
        json << "      },\n";

        // Write scan-level metadata if the E57 contains a Data3D block.
        // This is the metadata needed to know the scanner pose, scanner centre, and point-cloud extents.
        if (hasScan)
        {
            json << "      \"data3d\": {\n";
            json << "        \"name\": " << q(scan.name) << ",\n";
            json << "        \"guid\": " << q(scan.guid) << ",\n";
            json << "        \"description\": " << q(scan.description) << ",\n";
            json << "        \"sensor_vendor\": " << q(scan.sensorVendor) << ",\n";
            json << "        \"sensor_model\": " << q(scan.sensorModel) << ",\n";
            json << "        \"sensor_serial_number\": " << q(scan.sensorSerialNumber) << ",\n";
            json << "        \"sensor_hardware_version\": " << q(scan.sensorHardwareVersion) << ",\n";
            json << "        \"sensor_software_version\": " << q(scan.sensorSoftwareVersion) << ",\n";
            json << "        \"sensor_firmware_version\": " << q(scan.sensorFirmwareVersion) << ",\n";
            json << "        \"point_count\": " << scan.pointCount << ",\n";
            json << "        \"row_max\": " << rowMax << ",\n";
            json << "        \"column_max\": " << columnMax << ",\n";
            json << "        \"groups_size\": " << groupsSize << ",\n";
            json << "        \"scanner_center_world_m\": ["
                 << scan.pose.translation.x << ", "
                 << scan.pose.translation.y << ", "
                 << scan.pose.translation.z << "],\n";

            json << "        \"scan_pose\": ";
            writeTransformJson(json, scan.pose, 8);
            json << ",\n";

            json << "        \"cartesian_box\": ";
            writeBoundsJson(json, scan.cartesianBounds, scan.pose, 8);
            json << ",\n";

            // Spherical bounds are useful for checking scanner range and angular coverage.
            json << "        \"spherical_bounds\": {\n";
            json << "          \"range_minimum\": " << scan.sphericalBounds.rangeMinimum << ",\n";
            json << "          \"range_maximum\": " << scan.sphericalBounds.rangeMaximum << ",\n";
            json << "          \"elevation_minimum\": " << scan.sphericalBounds.elevationMinimum << ",\n";
            json << "          \"elevation_maximum\": " << scan.sphericalBounds.elevationMaximum << ",\n";
            json << "          \"azimuth_start\": " << scan.sphericalBounds.azimuthStart << ",\n";
            json << "          \"azimuth_end\": " << scan.sphericalBounds.azimuthEnd << "\n";
            json << "        }\n";
            json << "      },\n";
        }
        else
        {
            json << "      \"data3d\": null,\n";
        }

        // Simple list of filenames for quick Python access.
        json << "      \"pose_images\": [";
        for (size_t i = 0; i < poseImageNames.size(); ++i)
        {
            if (i > 0) json << ", ";
            json << q(poseImageNames[i]);
        }
        json << "],\n";

        // Detailed image metadata, including camera pose and projection model.
        json << "      \"images\": [\n";
        for (size_t i = 0; i < savedImages.size(); ++i)
        {
            if (i > 0) json << ",\n";
            writeImageJson(
                json,
                savedImages[i].header,
                savedImages[i].fileName,
                savedImages[i].projection,
                savedImages[i].imageType,
                savedImages[i].width,
                savedImages[i].height,
                savedImages[i].size,
                8);
        }
        json << "\n      ],\n";

        // Store warnings in JSON so problems are visible from Python without reading console logs.
        json << "      \"warnings\": [";
        bool wroteWarning = false;

        if (data3DCount == 0)
        {
            json << q("No Data3D block found in this E57.");
            wroteWarning = true;
        }

        if (data3DCount > 1)
        {
            if (wroteWarning) json << ", ";
            json << q("More than one Data3D block found. Only Data3D index 0 was used as the pose scan.");
            wroteWarning = true;
        }

        if (image2DCount > MAX_IMAGES_PER_POSE)
        {
            if (wroteWarning) json << ", ";
            json << q("More than 6 Image2D blocks found. Only the first 6 matching images were exported.");
            wroteWarning = true;
        }

        json << "]\n";
        json << "    }";
    }

    json << "\n  ]\n";
    json << "}\n";
    json.close();

    std::cout << "\nDone.\n";
    std::cout << "Pose JSON: " << jsonPath << "\n";
    std::cout << "Pose images folder: " << imagesFolder << "\n";

    return 0;
}
