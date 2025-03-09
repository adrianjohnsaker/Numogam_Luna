import android.os.Bundle;
import android.os.AsyncTask;
import android.widget.Toast;
import androidx.appcompat.app.AppCompatActivity;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

public class ModuleDownloader extends AppCompatActivity {
    
    private static final String REPO_BASE_URL = "https://raw.githubusercontent.com/your-username/numogram-luna/main/";
    private static final String MANIFEST_URL = REPO_BASE_URL + "modules/modules_manifest.json";
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_module_downloader);
        
        // Start downloading modules from manifest
        new DownloadModulesTask().execute();
    }
    
    private class DownloadModulesTask extends AsyncTask<Void, String, Boolean> {
        @Override
        protected Boolean doInBackground(Void... voids) {
            try {
                // First download the manifest to get the list of available modules
                String manifestJson = downloadContent(MANIFEST_URL);
                JSONObject manifest = new JSONObject(manifestJson);
                JSONArray modules = manifest.getJSONArray("modules");
                
                // Create modules directory if it doesn't exist
                File moduleDir = new File(getFilesDir(), "modules");
                if (!moduleDir.exists()) {
                    moduleDir.mkdirs();
                }
                
                // Download each module
                for (int i = 0; i < modules.length(); i++) {
                    JSONObject moduleInfo = modules.getJSONObject(i);
                    String moduleName = moduleInfo.getString("name");
                    String version = moduleInfo.getString("version");
                    
                    publishProgress("Downloading " + moduleName + " v" + version);
                    
                    // Download the module
                    String moduleUrl = REPO_BASE_URL + "modules/" + moduleName + ".py";
                    String moduleContent = downloadContent(moduleUrl);
                    
                    // Save the module
                    File moduleFile = new File(moduleDir, moduleName + ".py");
                    FileOutputStream outputStream = new FileOutputStream(moduleFile);
                    outputStream.write(moduleContent.getBytes());
                    outputStream.close();
                    
                    publishProgress("Installed " + moduleName);
                }
                
                return true;
            } catch (Exception e) {
                e.printStackTrace();
                publishProgress("Error: " + e.getMessage());
                return false;
            }
        }
        
        @Override
        protected void onProgressUpdate(String... values) {
            for (String message : values) {
                Toast.makeText(ModuleDownloader.this, message, Toast.LENGTH_SHORT).show();
            }
        }
        
        @Override
        protected void onPostExecute(Boolean success) {
            if (success) {
                Toast.makeText(ModuleDownloader.this, 
                        "All modules downloaded successfully", Toast.LENGTH_LONG).show();
            } else {
                Toast.makeText(ModuleDownloader.this, 
                        "Failed to download all modules", Toast.LENGTH_LONG).show();
            }
        }
    }
    
    private String downloadContent(String urlString) throws Exception {
        URL url = new URL(urlString);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        connection.setRequestMethod("GET");
        connection.setConnectTimeout(15000);
        connection.setReadTimeout(15000);
        connection.connect();
        
        int responseCode = connection.getResponseCode();
        if (responseCode != HttpURLConnection.HTTP_OK) {
            throw new Exception("HTTP error code: " + responseCode);
        }
        
        BufferedReader reader = new BufferedReader(
                new InputStreamReader(connection.getInputStream()));
        StringBuilder content = new StringBuilder();
        String line;
        
        while ((line = reader.readLine()) != null) {
            content.append(line).append("\n");
        }
        
        reader.close();
        connection.disconnect();
        
        return content.toString();
    }
}
